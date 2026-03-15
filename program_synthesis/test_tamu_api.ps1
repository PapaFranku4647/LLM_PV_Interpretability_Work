param(
    [string]$ApiKey = $env:TAMUS_AI_CHAT_API_KEY,
    [string]$ApiEndpoint = $env:TAMUS_AI_CHAT_API_ENDPOINT,
    [string]$Model = "protected.gemini-2.0-flash-lite",
    [ValidateSet("api", "openai", "auto")]
    [string]$Mode = "auto",
    [string]$Prompt = "Reply with exactly: TAMU API OK",
    [string]$ReasoningEffort = "",
    [int]$MaxTokens = 32
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

function Invoke-TamuChatCompletion {
    param(
        [string]$Uri,
        [hashtable]$Headers,
        [hashtable]$Payload,
        [string]$Label
    )

    $jsonBody = $Payload | ConvertTo-Json -Depth 10
    Write-Host "Trying $Label ..."
    Write-Host "Request body: $jsonBody"

    try {
        $response = Invoke-RestMethod `
            -Uri $Uri `
            -Headers $Headers `
            -Method Post `
            -Body $jsonBody

        $reply = $null
        if ($response.choices -and $response.choices.Count -gt 0) {
            $reply = $response.choices[0].message.content
        }

        return [PSCustomObject]@{
            ok = $true
            label = $Label
            model = $response.model
            reply = $reply
            prompt_tokens = $response.usage.prompt_tokens
            completion_tokens = $response.usage.completion_tokens
            total_tokens = $response.usage.total_tokens
            error_text = $null
        }
    } catch {
        $statusCode = $null
        if ($_.Exception.Response) {
            try {
                $statusCode = [int]$_.Exception.Response.StatusCode
            } catch {
                $statusCode = $null
            }
        }
        $errorText = Get-ErrorResponseText -Exception $_.Exception
        Write-Warning "Request '$Label' failed with status $statusCode"
        if ($errorText) {
            Write-Warning "Server response: $errorText"
        } else {
            Write-Warning "No response body was returned."
        }

        return [PSCustomObject]@{
            ok = $false
            label = $Label
            model = $Payload.model
            reply = $null
            prompt_tokens = $null
            completion_tokens = $null
            total_tokens = $null
            error_text = $errorText
        }
    }
}

function Get-ModeConfig {
    param(
        [string]$BaseEndpoint,
        [string]$SelectedMode
    )

    $trimmed = $BaseEndpoint.TrimEnd("/")
    switch ($SelectedMode) {
        "api" {
            return [PSCustomObject]@{
                Name = "api"
                ModelsUri = "$trimmed/api/models"
                ChatUri = "$trimmed/api/chat/completions"
            }
        }
        "openai" {
            return [PSCustomObject]@{
                Name = "openai"
                ModelsUri = "$trimmed/openai/models"
                ChatUri = "$trimmed/openai/chat/completions"
            }
        }
        default {
            throw "Unknown TAMU mode '$SelectedMode'."
        }
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

function New-ChatPayload {
    param(
        [string]$ModelName,
        [string]$MessageText,
        [bool]$IncludeStreamFlag,
        [bool]$IncludeMaxTokens,
        [int]$TokenLimit,
        [string]$Reasoning
    )

    $payload = @{
        model = $ModelName
        messages = @(
            @{
                role = "user"
                content = $MessageText
            }
        )
    }
    if ($IncludeStreamFlag) {
        $payload.stream = $false
    }
    if ($IncludeMaxTokens) {
        $payload.max_tokens = $TokenLimit
    }
    if ($Reasoning -and (Test-ModelSupportsReasoningEffort -ModelName $ModelName)) {
        $payload.reasoning_effort = $Reasoning
    }
    return $payload
}

if (-not $ApiEndpoint) {
    $ApiEndpoint = "https://chat-api.tamu.ai"
}
if (-not $ApiKey) {
    throw "TAMUS_AI_CHAT_API_KEY is missing."
}

$headers = @{
    Authorization = "Bearer $ApiKey"
    "Content-Type" = "application/json"
}

$modeOrder = if ($Mode -eq "auto") { @("api", "openai") } else { @($Mode) }
$modeConfig = $null
$modelsRaw = $null

foreach ($modeName in $modeOrder) {
    $candidate = Get-ModeConfig -BaseEndpoint $ApiEndpoint -SelectedMode $modeName
    Write-Host "Listing TAMU models from $($candidate.ModelsUri) ..."
    try {
        $modelsRaw = Invoke-RestMethod -Uri $candidate.ModelsUri -Headers $headers -Method Get
        $modeConfig = $candidate
        break
    } catch {
        $statusCode = $null
        if ($_.Exception.Response) {
            try {
                $statusCode = [int]$_.Exception.Response.StatusCode
            } catch {
                $statusCode = $null
            }
        }
        $errorText = Get-ErrorResponseText -Exception $_.Exception
        Write-Warning "Model listing failed for mode '$modeName' with status $statusCode"
        if ($errorText) {
            Write-Warning "Server response: $errorText"
        } else {
            Write-Warning "No response body was returned."
        }
    }
}

if (-not $modeConfig) {
    throw "Failed to list TAMU models via both '/api' and '/openai' paths."
}

$modelIds = @()
if ($modelsRaw -is [System.Collections.IEnumerable] -and -not ($modelsRaw -is [string])) {
    foreach ($item in $modelsRaw) {
        if ($null -ne $item.id) {
            $modelIds += [string]$item.id
        } elseif ($null -ne $item.name) {
            $modelIds += [string]$item.name
        }
    }
}
if ($modelsRaw.data) {
    foreach ($item in $modelsRaw.data) {
        if ($null -ne $item.id) {
            $modelIds += [string]$item.id
        } elseif ($null -ne $item.name) {
            $modelIds += [string]$item.name
        }
    }
}

$modelIds = $modelIds | Sort-Object -Unique
if ($modelIds.Count -gt 0) {
    Write-Host "Found $($modelIds.Count) models. First 20:"
    $modelIds | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
    if ($modelIds -notcontains $Model) {
        $fallback = $modelIds | Where-Object { $_ -like "protected.gemini*" } | Select-Object -First 1
        if (-not $fallback) {
            $fallback = $modelIds | Select-Object -First 1
        }
        Write-Warning "Requested model '$Model' was not found. Falling back to '$fallback'."
        $Model = $fallback
    }
}

Write-Host ""
Write-Host "Using TAMU mode '$($modeConfig.Name)'."
Write-Host "Sending a minimal chat completion to $Model ..."
$chatUri = $modeConfig.ChatUri
$payloads = @(
    @{
        label = "docs-minimal"
        body = New-ChatPayload -ModelName $Model -MessageText $Prompt -IncludeStreamFlag $true -IncludeMaxTokens $false -TokenLimit $MaxTokens -Reasoning $ReasoningEffort
    },
    @{
        label = "minimal-no-stream"
        body = New-ChatPayload -ModelName $Model -MessageText $Prompt -IncludeStreamFlag $false -IncludeMaxTokens $false -TokenLimit $MaxTokens -Reasoning $ReasoningEffort
    },
    @{
        label = "minimal-with-max-tokens"
        body = New-ChatPayload -ModelName $Model -MessageText $Prompt -IncludeStreamFlag $true -IncludeMaxTokens $true -TokenLimit $MaxTokens -Reasoning $ReasoningEffort
    }
)

$result = $null
foreach ($payload in $payloads) {
    $result = Invoke-TamuChatCompletion -Uri $chatUri -Headers $headers -Payload $payload.body -Label $payload.label
    if ($result.ok) {
        break
    }
    Write-Host ""
}

if (-not $result.ok) {
    throw "All TAMU chat completion attempts failed. Review the server response above."
}

$result | Select-Object label, model, reply, prompt_tokens, completion_tokens, total_tokens | Format-List
