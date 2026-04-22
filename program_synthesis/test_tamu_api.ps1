param(
    [string]$ApiKey = $(if ($env:TAMU_API_KEY) { $env:TAMU_API_KEY } else { $env:TAMUS_AI_CHAT_API_KEY }),
    [Alias("ApiEndpoint")]
    [string]$AzureEndpoint = $(if ($env:TAMU_AZURE_ENDPOINT) { $env:TAMU_AZURE_ENDPOINT } elseif ($env:DPF_URL) { $env:DPF_URL } else { "https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/" }),
    [string]$ApiVersion = $(if ($env:TAMU_API_VERSION) { $env:TAMU_API_VERSION } else { "2024-12-01-preview" }),
    [string]$Model = $(if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-5.2-deep-learning-fundamentals" }),
    [string]$Prompt = "Reply with exactly: TAMU API OK",
    [string]$ReasoningEffort = "minimal",
    [int]$MaxTokens = 32
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

if (-not $ApiKey) {
    throw "Missing API key. Set TAMU_API_KEY or TAMUS_AI_CHAT_API_KEY."
}

$requestUri = "$($AzureEndpoint.TrimEnd('/'))/openai/deployments/$Model/chat/completions?api-version=$ApiVersion"
$headers = @{
    "api-key" = $ApiKey
    "Content-Type" = "application/json"
}

$payload = @{
    messages = @(
        @{
            role = "user"
            content = $Prompt
        }
    )
    max_completion_tokens = $MaxTokens
}

if ($ReasoningEffort -and (Test-ModelSupportsReasoningEffort -ModelName $Model)) {
    $payload.reasoning_effort = $ReasoningEffort
}

Write-Host "Request URI: $requestUri"
Write-Host "Request body:"
$payload | ConvertTo-Json -Depth 20

$response = Invoke-RestMethod `
    -Uri $requestUri `
    -Headers $headers `
    -Method Post `
    -Body ($payload | ConvertTo-Json -Depth 20)

$reply = $null
if ($response.choices -and $response.choices.Count -gt 0) {
    $reply = Get-TextContent -Content $response.choices[0].message.content
}

$pricingModelName = if ($response.model) { $response.model } else { $Model }
$pricing = Get-EstimatedCost -ModelName $pricingModelName -PromptTokens $response.usage.prompt_tokens -CompletionTokens $response.usage.completion_tokens

[PSCustomObject]@{
    deployment = $Model
    returned_model = $response.model
    reply = $reply
    prompt_tokens = $response.usage.prompt_tokens
    completion_tokens = $response.usage.completion_tokens
    total_tokens = $response.usage.total_tokens
    reasoning_tokens = $(if ($response.usage.completion_tokens_details) { $response.usage.completion_tokens_details.reasoning_tokens } else { $null })
    pricing_model = $pricing.pricing_model
    input_rate_per_million = $pricing.input_rate_per_million
    output_rate_per_million = $pricing.output_rate_per_million
    estimated_input_cost_usd = $pricing.estimated_input_cost_usd
    estimated_output_cost_usd = $pricing.estimated_output_cost_usd
    estimated_total_cost_usd = $pricing.estimated_total_cost_usd
} | Format-List
