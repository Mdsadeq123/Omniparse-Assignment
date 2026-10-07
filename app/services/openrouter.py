from app.config import settings


def get_openrouter_headers() -> dict[str, str]:
    """Optional OpenRouter headers for leaderboard attribution."""
    headers: dict[str, str] = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_title:
        headers["X-Title"] = settings.openrouter_app_title
    return headers


def get_openrouter_client_kwargs() -> dict:
    return {
        "openai_api_key": settings.openrouter_api_key,
        "openai_api_base": settings.openrouter_base_url,
        "default_headers": get_openrouter_headers(),
    }
