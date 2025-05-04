from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

console = Console()

class ErrorHandler:
    @staticmethod
    def handle_api_error(error: Exception, context: str):
        """
        Handle API-related errors with user-friendly messages
        """
        error_panel = Panel(
            Text(
                f"❌ API Error: {str(error)}\n\n"
                f"Context: {context}\n"
                f"Please check your configuration and try again.",
                style="red"
            ),
            title="API Error",
            border_style="red"
        )
        console.print(error_panel)

    @staticmethod
    def show_loading(message: str):
        """
        Display a loading spinner with message
        """
        with console.status(f"[cyan]{message}..."):
            yield

    @staticmethod
    def validate_config(config: dict):
        """
        Validate required configuration settings
        """
        required_fields = ['base_url', 'provider']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            error_panel = Panel(
                Text(
                    f"❌ Missing required configuration fields: {', '.join(missing_fields)}\n\n"
                    "Please update your configuration file.",
                    style="red"
                ),
                title="Configuration Error",
                border_style="red"
            )
            console.print(error_panel)
            return False
        return True

    @staticmethod
    def show_success(message: str):
        """
        Display a success message
        """
        success_panel = Panel(
            Text(
                f"✓ {message}",
                style="green"
            ),
            title="Success",
            border_style="green"
        )
        console.print(success_panel)
