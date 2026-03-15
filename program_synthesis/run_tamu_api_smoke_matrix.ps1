param(
    [string]$ApiKey = $env:TAMUS_AI_CHAT_API_KEY,
    [string]$ApiEndpoint = $env:TAMUS_AI_CHAT_API_ENDPOINT,
    [string[]]$Models = @(
        "protected.gpt-5",
        "protected.gemini-2.0-flash-lite",
        "protected.gemini-2.5-flash-lite"
    ),
    [string[]]$ReasoningEfforts = @("minimal", "medium"),
    [ValidateSet("quick", "standard")]
    [string]$PromptSet = "standard",
    [int]$MaxTokens = 512,
    [int]$PauseBetweenCallsMs = 0,
    [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"

function Get-ErrorResponseText {
    param([System.Exception]$Exception)

    $response = $Exception.Response
    if ($null -eq $response) {
        return $null
    }

    try {
        $stream = $response.GetResponseStream()
        if ($null -eq $stream) {
            return $null
        }
        $reader = New-Object System.IO.StreamReader($stream)
        $text = $reader.ReadToEnd()
        $reader.Close()
        return $text
    } catch {
        return $null
    }
}

function Test-ModelSupportsReasoningEffort {
    param([string]$ModelName)

    $lowered = $ModelName.ToLowerInvariant()
    return (
        $lowered.StartsWith("o3") -or
        $lowered.StartsWith("o4") -or
        $lowered.Contains("gpt-5") -or
        $lowered.StartsWith("openai/o3") -or
        $lowered.StartsWith("openai/o4")
    )
}

function Get-AvailableModelIds {
    param([object]$ModelsResponse)

    $modelIds = @()
    if ($ModelsResponse -is [System.Collections.IEnumerable] -and -not ($ModelsResponse -is [string])) {
        foreach ($item in $ModelsResponse) {
            if ($null -ne $item.id) {
                $modelIds += [string]$item.id
            } elseif ($null -ne $item.name) {
                $modelIds += [string]$item.name
            }
        }
    }
    if ($ModelsResponse.data) {
        foreach ($item in $ModelsResponse.data) {
            if ($null -ne $item.id) {
                $modelIds += [string]$item.id
            } elseif ($null -ne $item.name) {
                $modelIds += [string]$item.name
            }
        }
    }
    return $modelIds | Sort-Object -Unique
}

function Get-TextContent {
    param([object]$Content)

    if ($null -eq $Content) {
        return ""
    }
    if ($Content -is [string]) {
        return $Content
    }
    if ($Content -is [System.Collections.IEnumerable] -and -not ($Content -is [string])) {
        $parts = foreach ($item in $Content) {
            if ($item.text) {
                [string]$item.text
            } elseif ($item.content) {
                Get-TextContent -Content $item.content
            }
        }
        return ($parts -join "")
    }
    if ($Content.text) {
        return [string]$Content.text
    }
    if ($Content.content) {
        return Get-TextContent -Content $Content.content
    }
    return [string]$Content
}

function New-PromptCases {
    param([string]$SelectedPromptSet)

    if ($SelectedPromptSet -eq "quick") {
        return @(
            @{
                label = "echo_ok"
                text = "Reply with exactly: TAMU API OK"
            }
        )
    }

    return @(
        @{
            label = "echo_ok"
            text = "Reply with exactly: TAMU API OK"
        },
        @{
            label = "why_sky_blue"
            text = "Why is the sky blue?"
        }
    )
}

if (-not $ApiEndpoint) {
    $ApiEndpoint = "https://chat-api.tamu.ai"
}
if (-not $ApiKey) {
    throw "TAMUS_AI_CHAT_API_KEY is missing."
}
if (-not $OutRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = "program_synthesis/runs_tamu_api_smoke/$stamp"
}

$apiRoot = $ApiEndpoint.TrimEnd("/")
$headers = @{
    Authorization = "Bearer $ApiKey"
    "Content-Type" = "application/json"
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$modelsPath = Join-Path $OutRoot "models.json"
$resultsJsonlPath = Join-Path $OutRoot "results.jsonl"
$resultsCsvPath = Join-Path $OutRoot "results.csv"
$manifestPath = Join-Path $OutRoot "manifest.json"

Write-Host "Listing models from $apiRoot/api/models ..."
$modelsResponse = Invoke-RestMethod -Uri "$apiRoot/api/models" -Headers $headers -Method Get
$modelsResponse | ConvertTo-Json -Depth 100 | Set-Content -Path $modelsPath -Encoding UTF8
$availableModels = Get-AvailableModelIds -ModelsResponse $modelsResponse

Write-Host "Found $($availableModels.Count) models."

$selectedModels = @()
foreach ($model in $Models) {
    if ($availableModels -contains $model) {
        $selectedModels += $model
    } else {
        Write-Warning "Skipping unavailable model '$model'."
    }
}

if ($selectedModels.Count -eq 0) {
    throw "None of the requested models were available."
}

$promptCases = New-PromptCases -SelectedPromptSet $PromptSet
$results = New-Object System.Collections.Generic.List[object]

foreach ($model in $selectedModels) {
    $efforts = @("")
    if (Test-ModelSupportsReasoningEffort -ModelName $model) {
        $efforts = $ReasoningEfforts
    }

    foreach ($effort in $efforts) {
        foreach ($promptCase in $promptCases) {
            $body = @{
                model = $model
                stream = $false
                max_tokens = $MaxTokens
                messages = @(
                    @{
                        role = "user"
                        content = $promptCase.text
                    }
                )
            }
            if ($effort) {
                $body.reasoning_effort = $effort
            }

            $startedAt = Get-Date
            $status = "ok"
            $responseModel = $null
            $finishReason = $null
            $reply = $null
            $promptTokens = $null
            $completionTokens = $null
            $totalTokens = $null
            $reasoningTokens = $null
            $errorText = $null

            $effortLabel = if ($effort) { $effort } else { "default" }

            Write-Host ""
            Write-Host "=== Request ==="
            Write-Host "model=$model prompt=$($promptCase.label) reasoning=$effortLabel"

            try {
                $response = Invoke-RestMethod `
                    -Uri "$apiRoot/api/chat/completions" `
                    -Headers $headers `
                    -Method Post `
                    -Body ($body | ConvertTo-Json -Depth 20)

                $responseModel = $response.model
                if ($response.choices -and $response.choices.Count -gt 0) {
                    $finishReason = $response.choices[0].finish_reason
                    $reply = Get-TextContent -Content $response.choices[0].message.content
                }
                if ($response.usage) {
                    $promptTokens = $response.usage.prompt_tokens
                    $completionTokens = $response.usage.completion_tokens
                    $totalTokens = $response.usage.total_tokens
                    if ($response.usage.completion_tokens_details) {
                        $reasoningTokens = $response.usage.completion_tokens_details.reasoning_tokens
                    }
                }
            } catch {
                $status = "error"
                $errorText = Get-ErrorResponseText -Exception $_.Exception
                if (-not $errorText) {
                    $errorText = $_.Exception.Message
                }
            }

            $elapsedMs = [int]((Get-Date) - $startedAt).TotalMilliseconds
            $replyPreview = if ($reply) {
                if ($reply.Length -gt 160) { $reply.Substring(0, 160) } else { $reply }
            } else {
                $null
            }

            $row = [PSCustomObject]@{
                ts = $startedAt.ToString("s")
                model = $model
                requested_reasoning_effort = $effortLabel
                prompt_label = $promptCase.label
                status = $status
                elapsed_ms = $elapsedMs
                returned_model = $responseModel
                finish_reason = $finishReason
                prompt_tokens = $promptTokens
                completion_tokens = $completionTokens
                total_tokens = $totalTokens
                reasoning_tokens = $reasoningTokens
                reply_preview = $replyPreview
                error = $errorText
            }
            $results.Add($row) | Out-Null
            $row | ConvertTo-Json -Depth 20 -Compress | Add-Content -Path $resultsJsonlPath -Encoding UTF8

            if ($PauseBetweenCallsMs -gt 0) {
                Start-Sleep -Milliseconds $PauseBetweenCallsMs
            }
        }
    }
}

$results | Export-Csv -Path $resultsCsvPath -NoTypeInformation -Encoding UTF8

$manifest = [PSCustomObject]@{
    out_root = $OutRoot
    api_endpoint = $ApiEndpoint
    models = $selectedModels
    reasoning_efforts = $ReasoningEfforts
    prompt_set = $PromptSet
    max_tokens = $MaxTokens
    pause_between_calls_ms = $PauseBetweenCallsMs
    results_csv = $resultsCsvPath
    results_jsonl = $resultsJsonlPath
    models_json = $modelsPath
}
$manifest | ConvertTo-Json -Depth 20 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "=== Summary ==="
$results |
    Select-Object model, requested_reasoning_effort, prompt_label, status, finish_reason, total_tokens, reasoning_tokens, elapsed_ms |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Artifacts:"
Write-Host "  models_json = $modelsPath"
Write-Host "  results_jsonl = $resultsJsonlPath"
Write-Host "  results_csv = $resultsCsvPath"
Write-Host "  manifest_json = $manifestPath"
