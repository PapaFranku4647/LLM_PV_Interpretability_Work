param(
    [string]$ApiKey = $(if ($env:TAMU_API_KEY) { $env:TAMU_API_KEY } else { $env:TAMUS_AI_CHAT_API_KEY }),
    [Alias("ApiEndpoint")]
    [string]$AzureEndpoint = $(if ($env:TAMU_AZURE_ENDPOINT) { $env:TAMU_AZURE_ENDPOINT } elseif ($env:DPF_URL) { $env:DPF_URL } else { "https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/" }),
    [string]$ApiVersion = $(if ($env:TAMU_API_VERSION) { $env:TAMU_API_VERSION } else { "2024-12-01-preview" }),
    [Alias("Models")]
    [string[]]$Deployments = @("gpt-5.2-deep-learning-fundamentals"),
    [string[]]$ReasoningEfforts = @("minimal"),
    [ValidateSet("quick", "standard")]
    [string]$PromptSet = "quick",
    [int]$MaxTokens = 64,
    [int]$PauseBetweenCallsMs = 0,
    [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"

function Test-ModelSupportsReasoningEffort {
    param([string]$ModelName)

    $lowered = [string]$ModelName
    if (-not $lowered) {
        $lowered = ""
    }
    $lowered = $lowered.ToLowerInvariant()
    return (
        $lowered.StartsWith("o3") -or
        $lowered.StartsWith("o4") -or
        $lowered.Contains("gpt-5") -or
        $lowered.StartsWith("openai/o3") -or
        $lowered.StartsWith("openai/o4")
    )
}

function Get-ModelPricing {
    param([string]$ModelName)

    $lowered = [string]$ModelName
    if (-not $lowered) {
        $lowered = ""
    }
    $lowered = $lowered.ToLowerInvariant()
    if ($lowered -match "gpt-5\.2") { return @{ name = "gpt-5.2"; input = 1.75; output = 14.00 } }
    if ($lowered -match "gpt-5\.1") { return @{ name = "gpt-5.1"; input = 1.25; output = 10.00 } }
    if ($lowered -match "gpt-5") { return @{ name = "gpt-5"; input = 1.25; output = 10.00 } }
    if ($lowered -match "gpt-4\.1") { return @{ name = "gpt-4.1"; input = 2.00; output = 8.00 } }
    if ($lowered -match "gpt-4o") { return @{ name = "gpt-4o"; input = 2.50; output = 10.00 } }
    if ($lowered -match "o3-mini") { return @{ name = "o3-mini"; input = 1.10; output = 4.40 } }
    if ($lowered -match "o4-mini") { return @{ name = "o4-mini"; input = 1.10; output = 4.40 } }
    return $null
}

function Get-EstimatedCost {
    param(
        [string]$ModelName,
        [object]$PromptTokens,
        [object]$CompletionTokens
    )

    $pricing = Get-ModelPricing -ModelName $ModelName
    if ($null -eq $pricing) {
        return [PSCustomObject]@{
            pricing_model = $null
            input_rate_per_million = $null
            output_rate_per_million = $null
            estimated_input_cost_usd = $null
            estimated_output_cost_usd = $null
            estimated_total_cost_usd = $null
        }
    }

    $prompt = 0.0
    $completion = 0.0
    if ($PromptTokens -ne $null -and "$PromptTokens" -ne "") { $prompt = [double]$PromptTokens }
    if ($CompletionTokens -ne $null -and "$CompletionTokens" -ne "") { $completion = [double]$CompletionTokens }
    $inputCost = ($prompt / 1000000.0) * [double]$pricing.input
    $outputCost = ($completion / 1000000.0) * [double]$pricing.output

    return [PSCustomObject]@{
        pricing_model = $pricing.name
        input_rate_per_million = [double]$pricing.input
        output_rate_per_million = [double]$pricing.output
        estimated_input_cost_usd = $inputCost
        estimated_output_cost_usd = $outputCost
        estimated_total_cost_usd = $inputCost + $outputCost
    }
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
            label = "small_reasoning"
            text = "In one sentence, explain why 2 + 2 = 4."
        }
    )
}

if (-not $ApiKey) {
    throw "Missing API key. Set TAMU_API_KEY or TAMUS_AI_CHAT_API_KEY."
}
if (-not $OutRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = "program_synthesis/runs_tamu_api_smoke/$stamp"
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$resultsJsonlPath = Join-Path $OutRoot "results.jsonl"
$resultsCsvPath = Join-Path $OutRoot "results.csv"
$manifestPath = Join-Path $OutRoot "manifest.json"

$promptCases = New-PromptCases -SelectedPromptSet $PromptSet
$results = New-Object System.Collections.Generic.List[object]
$headers = @{
    "api-key" = $ApiKey
    "Content-Type" = "application/json"
}

foreach ($deployment in $Deployments) {
    $efforts = @("")
    if (Test-ModelSupportsReasoningEffort -ModelName $deployment) {
        $efforts = $ReasoningEfforts
    }

    foreach ($effort in $efforts) {
        foreach ($promptCase in $promptCases) {
            $body = @{
                messages = @(
                    @{
                        role = "user"
                        content = $promptCase.text
                    }
                )
                max_completion_tokens = $MaxTokens
            }
            if ($effort) {
                $body.reasoning_effort = $effort
            }

            $requestUri = "$($AzureEndpoint.TrimEnd('/'))/openai/deployments/$deployment/chat/completions?api-version=$ApiVersion"
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
            Write-Host "deployment=$deployment prompt=$($promptCase.label) reasoning=$effortLabel"

            try {
                $response = Invoke-RestMethod `
                    -Uri $requestUri `
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
                $errorText = $_.Exception.Message
            }

            $elapsedMs = [int]((Get-Date) - $startedAt).TotalMilliseconds
            $replyPreview = if ($reply) {
                if ($reply.Length -gt 160) { $reply.Substring(0, 160) } else { $reply }
            } else {
                $null
            }
            $pricingModelName = if ($responseModel) { $responseModel } else { $deployment }
            $pricing = Get-EstimatedCost -ModelName $pricingModelName -PromptTokens $promptTokens -CompletionTokens $completionTokens

            $row = [PSCustomObject]@{
                ts = $startedAt.ToString("s")
                deployment = $deployment
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
                pricing_model = $pricing.pricing_model
                input_rate_per_million = $pricing.input_rate_per_million
                output_rate_per_million = $pricing.output_rate_per_million
                estimated_input_cost_usd = $pricing.estimated_input_cost_usd
                estimated_output_cost_usd = $pricing.estimated_output_cost_usd
                estimated_total_cost_usd = $pricing.estimated_total_cost_usd
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
    azure_endpoint = $AzureEndpoint
    api_version = $ApiVersion
    deployments = $Deployments
    reasoning_efforts = $ReasoningEfforts
    prompt_set = $PromptSet
    max_tokens = $MaxTokens
    pause_between_calls_ms = $PauseBetweenCallsMs
    results_csv = $resultsCsvPath
    results_jsonl = $resultsJsonlPath
}
$manifest | ConvertTo-Json -Depth 20 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "=== Summary ==="
$results |
    Select-Object deployment, requested_reasoning_effort, prompt_label, status, finish_reason, total_tokens, reasoning_tokens, estimated_total_cost_usd, elapsed_ms |
    Format-Table -AutoSize

$totalEstimatedCost = ($results | Measure-Object -Property estimated_total_cost_usd -Sum).Sum
$totalEstimatedCostValue = if ($null -ne $totalEstimatedCost) { [double]$totalEstimatedCost } else { 0.0 }
Write-Host ""
Write-Host ("Estimated total cost (USD): {0:N6}" -f $totalEstimatedCostValue)

Write-Host ""
Write-Host "Artifacts:"
Write-Host "  results_jsonl = $resultsJsonlPath"
Write-Host "  results_csv = $resultsCsvPath"
Write-Host "  manifest_json = $manifestPath"
