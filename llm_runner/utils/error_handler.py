from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from typing import Optional

class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass

class ConfigurationError(LLMError):
    """Raised when there's a configuration issue"""
    pass

class APIError(LLMError):
    """Raised when there's an API-related error"""
    def __init__(self, message: str, context: Optional[str] = None):
        super().__init__(message)
        self.context = context

class ErrorHandler:
    @staticmethod
    def handle_error(error: Exception, context: Optional[str] = None):
        """Handle errors with user-friendly messages"""
        console = Console()
        error_type = type(error).__name__
        
        if isinstance(error, ConfigurationError):
            ErrorHandler._show_config_error(error, context)
        elif isinstance(error, APIError):
            ErrorHandler._show_api_error(error, context)
        else:
            ErrorHandler._show_generic_error(error, context)
            
    @staticmethod
    def _show_config_error(error: ConfigurationError, context: Optional[str]):
        """Show configuration error message"""
        error_panel = Panel(
            Text(
                f"❌ Configuration Error: {str(error)}\n\n"
                f"Context: {context or 'Configuration'}\n"
                f"Please check your configuration and try again.",
                style="red"
            ),
            title="Configuration Error",
            border_style="red"
        )
        console.print(error_panel)

    @staticmethod
    def _show_api_error(error: APIError, context: Optional[str]):
        """Show API error message"""
        error_panel = Panel(
            Text(
                f"❌ API Error: {str(error)}\n\n"
                f"Context: {context or 'API Request'}\n"
                f"Please check your connection and try again.",
                style="red"
            ),
            title="API Error",
            border_style="red"
        )
        console.print(error_panel)

    @staticmethod
    def _show_generic_error(error: Exception, context: Optional[str]):
        """Show generic error message"""
        error_panel = Panel(
            Text(
                f"❌ Error: {str(error)}\n\n"
                f"Context: {context or 'Unknown'}\n"
                f"Please check the logs for more details.",
                style="red"
            ),
            title="Error",
            border_style="red"
        )
        console.print(error_panel)

    @staticmethod
    def show_loading(message: str):
        """Display a loading spinner with message"""
        console = Console()
        with console.status(f"[cyan]{message}..."):
            yield

    @staticmethod
    def show_success(message: str):
        """Display a success message"""
        console = Console()
        success_panel = Panel(
            Text(
                f"✓ {message}",
                style="green"
            ),
            title="Success",
            border_style="green"
        )
        console.print(success_panel)
