import argparse
import time
import os
import json
from typing import List, Optional, Tuple, Dict, Any
import ast

from openai import OpenAI
from bs4 import BeautifulSoup
from ebooklib import epub
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


class TranslationService:
    """Base class for translation services"""
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate a batch of texts from source_lang to target_lang"""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAITranslator(TranslationService):
    """Translation service using OpenAI's API"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", rate_limit_wait: int = 3, 
                model_config: Optional[Dict] = None):
        """
        Initialize the OpenAI translator
        
        Args:
            api_key: OpenAI API key
            model: Model to use for translation
            rate_limit_wait: Seconds to wait between API calls to avoid rate limiting
            model_config: Configuration details for available models
        """
        self.api_key = api_key
        self.model = model
        self.model_config = model_config or {}
        
        # Get model-specific rate limit wait time if available
        if model_config and model in model_config:
            self.rate_limit_wait = model_config[model].get("rate_limit_wait", rate_limit_wait)
        else:
            self.rate_limit_wait = rate_limit_wait
            
        self.client = OpenAI(api_key=api_key)
        self.console = Console()
        
        # Log the model being used
        model_info = ""
        if model_config and model in model_config:
            model_info = f" - {model_config[model].get('description', '')}"
        self.console.print(f"[cyan]Using OpenAI model:[/cyan] [bold]{model}[/bold]{model_info}")
        
    def translate_batch(self, texts: List[str], source_lang: str = "auto", target_lang: str = "Chinese") -> List[str]:
        """
        Translate a batch of texts using OpenAI
        
        Args:
            texts: List of texts to translate
            source_lang: Source language (default: auto-detect)
            target_lang: Target language (default: Chinese)
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
            
        self.console.print(f"Translating batch of {len(texts)} texts to {target_lang}")
        
        # Format the input for better context
        input_text = str(texts)
        
        # Create system message based on languages
        system_message = f"You are a professional translator from {source_lang} to {target_lang}."
        
        try:
            response = self._make_translation_request(system_message, input_text, target_lang)
            translated_texts = self._parse_translation_response(response)
            
            # Verify we got the right number of translations
            if len(translated_texts) != len(texts):
                self.console.print(f"[yellow]Warning: Expected {len(texts)} translations but got {len(translated_texts)}. Retrying...[/yellow]")
                # If length mismatch, try again with clearer instructions
                response = self._make_translation_request(
                    system_message, 
                    input_text, 
                    target_lang,
                    "Return exactly the same number of elements as in the input list. Each element should be a translation of the corresponding element in the input."
                )
                translated_texts = self._parse_translation_response(response)
                
                # If still mismatch, pad or truncate to match (last resort)
                if len(translated_texts) != len(texts):
                    self.console.print(f"[red]Warning: Translation count mismatch. Adjusting output.[/red]")
                    if len(translated_texts) < len(texts):
                        # Pad with empty strings
                        translated_texts.extend([""] * (len(texts) - len(translated_texts)))
                    else:
                        # Truncate
                        translated_texts = translated_texts[:len(texts)]
                        
            # Wait to avoid rate limits
            time.sleep(self.rate_limit_wait)
            return translated_texts
            
        except Exception as e:
            self.console.print(f"[red]Translation error: {str(e)}[/red]")
            self.console.print("[yellow]Waiting 60 seconds due to potential rate limiting...[/yellow]")
            time.sleep(60)  # Longer wait after error
            
            # Retry once after error
            try:
                response = self._make_translation_request(system_message, input_text, target_lang)
                translated_texts = self._parse_translation_response(response)
                return translated_texts
            except Exception as retry_error:
                self.console.print(f"[red]Retry failed: {str(retry_error)}[/red]")
                # Return empty translations as fallback
                return ["[Translation failed]"] * len(texts)
    
    def _make_translation_request(self, system_message: str, input_text: str, target_lang: str, 
                                 additional_instructions: str = "") -> Dict[str, Any]:
        """Make the actual API request to OpenAI"""
        prompt = (f"Please translate the following text to {target_lang}. "
                 f"Return only the translated content as a list in the same format as the input. "
                 f"Maintain the same formatting as the original text for each list element. "
                 f"{additional_instructions}")
                 
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{prompt} ```{input_text}```"}
            ],
            temperature=0.3  # Lower temperature for more consistent translations
        )
        
        return response
        
    def _parse_translation_response(self, response) -> List[str]:
        """Parse the response from OpenAI into a list of translated texts"""
        # Extract the translated text from the response
        t_text = response.choices[0].message.content.strip()
        
        # Try to parse as a Python list if it looks like one
        if t_text.startswith('[') and t_text.endswith(']'):
            try:
                t_text = ast.literal_eval(t_text)
                if isinstance(t_text, list):
                    return t_text
            except (SyntaxError, ValueError):
                # Not a valid Python list, continue with text processing
                pass
        
        # If it's not a valid list or doesn't look like one, 
        # try to split by newlines as fallback
        if not isinstance(t_text, list):
            # Remove markdown code blocks if present
            if t_text.startswith('```') and t_text.endswith('```'):
                t_text = t_text[3:-3].strip()
            return t_text.split('\n')
        
        return t_text


class EPUBProcessor:
    """Process EPUB files for translation"""
    
    def __init__(self, epub_path: str, translator: TranslationService, batch_size: int = 5, 
                translation_only: bool = False):
        """
        Initialize the EPUB processor
        
        Args:
            epub_path: Path to the EPUB file
            translator: Translation service to use
            batch_size: Number of paragraphs to translate at once
            translation_only: If True, only show translation without the original text
        """
        self.epub_path = epub_path
        self.translator = translator
        self.batch_size = min(max(1, batch_size), 10)  # Ensure between 1 and 10
        self.translation_only = translation_only
        self.console = Console()
        
    def translate_epub(self, source_lang: str = "auto", target_lang: str = "Chinese", 
                      output_path: Optional[str] = None) -> str:
        """
        Translate the EPUB file
        
        Args:
            source_lang: Source language (default: auto-detect)
            target_lang: Target language (default: Chinese)
            output_path: Path to save the translated EPUB (default: auto-generated)
            
        Returns:
            Path to the translated EPUB file
        """
        if not output_path:
            # Generate output path based on input path
            base_name = os.path.splitext(self.epub_path)[0]
            output_path = f"{base_name}_{target_lang.lower()}.epub"
        
        self.console.print(f"[bold green]Translating[/bold green] {self.epub_path} → {output_path}")
        self.console.print(f"From: {source_lang} to: {target_lang} (Batch size: {self.batch_size})")
        self.console.print(f"Translation mode: {'Translation only' if self.translation_only else 'Original + Translation'}")
        
        # Read the original book
        origin_book = epub.read_epub(self.epub_path)
        
        # Create a new book with the same metadata
        new_book = epub.EpubBook()
        new_book.metadata = origin_book.metadata
        new_book.spine = origin_book.spine
        new_book.toc = origin_book.toc
        
        # Count total paragraphs for progress tracking
        total_paragraphs = 0
        html_items = [item for item in origin_book.get_items() if item.get_type() == 9]
        
        for item in html_items:
            soup = BeautifulSoup(item.content, "html.parser")
            p_tags = soup.find_all("p")
            total_paragraphs += len([p for p in p_tags if p.text and not p.text.isdigit()])
        
        # Process each item in the book
        translated_paragraphs = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"Translating EPUB to {target_lang}...", total=total_paragraphs)
            
            for item in origin_book.get_items():
                # Only process HTML content
                if item.get_type() == 9:
                    item = self._process_html_item(item, source_lang, target_lang, progress, task)
                    translated_paragraphs += 1
                
                # Add the item to the new book
                new_book.add_item(item)
        
        # Write the new book
        epub.write_epub(output_path, new_book, {})
        
        self.console.print(f"[bold green]Translation complete![/bold green] Saved to: {output_path}")
        self.console.print(f"Translated {translated_paragraphs} paragraphs across {len(html_items)} HTML files")
        
        return output_path
    
    def _process_html_item(self, item, source_lang, target_lang, progress, task):
        """Process an HTML item from the EPUB file"""
        soup = BeautifulSoup(item.content, "html.parser")
        p_tags = soup.find_all("p")
        
        # Skip processing if no paragraphs
        if not p_tags:
            return item
            
        batch = []
        
        for p in p_tags:
            # Skip empty paragraphs or those only containing numbers
            if not p.text or p.text.isdigit():
                continue
                
            batch.append(p)
            
            # Process batch when it reaches the batch size
            if len(batch) >= self.batch_size:
                self._translate_batch(batch, source_lang, target_lang)
                progress.update(task, advance=len(batch))
                batch = []
        
        # Process remaining paragraphs
        if batch:
            self._translate_batch(batch, source_lang, target_lang)
            progress.update(task, advance=len(batch))
        
        # Update the item content
        item.content = soup.prettify().encode()
        return item
    
    def _translate_batch(self, paragraphs, source_lang, target_lang):
        """Translate a batch of paragraphs and update their content"""
        # Extract text from paragraphs
        texts = [p.text for p in paragraphs]
        
        # Translate texts
        translated_texts = self.translator.translate_batch(texts, source_lang, target_lang)
        
        # Update paragraphs with translations
        for i, p in enumerate(paragraphs):
            if i < len(translated_texts):
                if self.translation_only:
                    # Replace original text with translation only
                    p.string = translated_texts[i]
                else:
                    # Add translation after original text
                    p.string = f"{p.text} [{translated_texts[i]}]"


def load_config(config_path="config.json") -> Dict:
    """Load configuration from JSON file"""
    # Check if config file exists
    if not os.path.exists(config_path):
        console = Console()
        
        # Check if template exists
        template_path = f"{config_path}.template"
        if os.path.exists(template_path):
            # Copy template to config file
            try:
                with open(template_path, 'r', encoding='utf-8') as template_file:
                    template_config = json.load(template_file)
                
                with open(config_path, 'w', encoding='utf-8') as config_file:
                    json.dump(template_config, config_file, indent=2)
                    
                console.print(f"[green]Created config file from template at {config_path}[/green]")
                console.print("[yellow]Please edit this file to add your OpenAI API key[/yellow]")
            except Exception as e:
                console.print(f"[red]Error creating config from template: {str(e)}[/red]")
                create_default_config(config_path)
        else:
            # Create default config
            create_default_config(config_path)
        
    # Load config file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        console = Console()
        console.print(f"[red]Error loading config file: {str(e)}[/red]")
        console.print("[yellow]Using default configuration[/yellow]")
        return get_default_config()


def create_default_config(config_path):
    """Create a default configuration file"""
    console = Console()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(get_default_config(), f, indent=2)
        console.print(f"[yellow]Created default configuration file at {config_path}[/yellow]")
        console.print("[yellow]Please edit this file to add your OpenAI API key[/yellow]")
    except Exception as e:
        console.print(f"[red]Error creating default config: {str(e)}[/red]")


def get_default_config() -> Dict:
    """Get default configuration dictionary"""
    return {
        "openai": {
            "api_key": "",
            "default_model": "gpt-4o-mini"
        },
        "translation": {
            "default_source_language": "auto",
            "default_target_language": "Chinese",
            "default_batch_size": 5,
            "translation_only": False
        },
        "available_models": {
            "gpt-4o-mini": {
                "description": "Fast and accurate for most translations",
                "context_length": 128000,
                "rate_limit_wait": 3
            },
            "gpt-4": {
                "description": "Higher accuracy for complex or nuanced content",
                "context_length": 8192,
                "rate_limit_wait": 5
            }
        }
    }


def list_available_models(config):
    """Print available models from the configuration"""
    console = Console()
    console.print("\n[bold cyan]Available Translation Models:[/bold cyan]")
    
    for model, details in config.get("available_models", {}).items():
        description = details.get("description", "No description available")
        context = details.get("context_length", "Unknown")
        console.print(f"[bold]{model}[/bold]: {description} (Context: {context} tokens)")
    
    console.print()


def main():
    # Load configuration
    config = load_config()
    
    # Get defaults from config
    default_model = config.get("openai", {}).get("default_model", "gpt-3.5-turbo")
    default_source = config.get("translation", {}).get("default_source_language", "auto")
    default_target = config.get("translation", {}).get("default_target_language", "Chinese")
    default_batch = config.get("translation", {}).get("default_batch_size", 5)
    translation_only = config.get("translation", {}).get("translation_only", False)
    config_api_key = config.get("openai", {}).get("api_key", "")
    
    # Get available models for choices
    available_models = list(config.get("available_models", {}).keys())
    if not available_models:
        available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    parser = argparse.ArgumentParser(description="Translate EPUB books using AI")
    parser.add_argument(
        "--book_path",
        dest="book_path",
        type=str,
        required=True,
        help="Path to the EPUB book file"
    )
    parser.add_argument(
        "--openai_key",
        dest="openai_key",
        type=str,
        default="",
        help="OpenAI API key (overrides config.json and environment variable)"
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=default_batch,
        choices=range(1, 11),
        help=f"Number of paragraphs to translate at once (1-10, default: {default_batch})"
    )
    parser.add_argument(
        "--source_lang",
        dest="source_lang",
        type=str,
        default=default_source,
        help=f"Source language (default: {default_source})"
    )
    parser.add_argument(
        "--target_lang",
        dest="target_lang",
        type=str,
        default=default_target,
        help=f"Target language (default: {default_target})"
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=str,
        default=None,
        help="Path to save the translated EPUB (default: auto-generated)"
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default=default_model,
        choices=available_models,
        help=f"OpenAI model to use (default: {default_model})"
    )
    parser.add_argument(
        "--translation_only",
        dest="translation_only",
        action="store_true",
        help="Show only the translated text without the original (default: False)"
    )
    parser.add_argument(
        "--list-models",
        dest="list_models",
        action="store_true",
        help="List available translation models and exit"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    options = parser.parse_args()
    
    # If just listing models, do that and exit
    if options.list_models:
        list_available_models(config)
        return
    
    # Validate inputs
    if not options.book_path.endswith(".epub"):
        raise ValueError("Input file must be an EPUB file (with .epub extension)")
    
    if not os.path.exists(options.book_path):
        raise FileNotFoundError(f"EPUB file not found: {options.book_path}")
    
    # API key precedence: command line > environment variable > config file
    api_key = options.openai_key or os.environ.get("OPENAI_API_KEY", "") or config_api_key
    
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Provide it using one of these methods:\n"
            "1. Command line: --openai_key YOUR_KEY\n"
            "2. Environment variable: OPENAI_API_KEY=YOUR_KEY\n"
            "3. Config file: Edit config.json with your API key"
        )
    
    # Get model configuration
    model_config = config.get("available_models", {})
    
    # Initialize translator and processor
    translator = OpenAITranslator(
        api_key=api_key, 
        model=options.model,
        model_config=model_config
    )
    
    # Use translation_only from command line if specified, otherwise from config
    translation_only_setting = options.translation_only if options.translation_only else translation_only
    
    processor = EPUBProcessor(
        epub_path=options.book_path,
        translator=translator,
        batch_size=options.batch_size,
        translation_only=translation_only_setting
    )
    
    # Translate the book
    processor.translate_epub(
        source_lang=options.source_lang,
        target_lang=options.target_lang,
        output_path=options.output_path
    )


if __name__ == "__main__":
    main()