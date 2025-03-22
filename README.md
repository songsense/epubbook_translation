## Running with Docker

This project can be run using Docker, which eliminates the need to install dependencies directly on your system.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/songsense/epub-translator.git
   cd epub-translator
   ```

2. Create your configuration file:
   ```
   cp config.json.template config.json
   ```
   Edit `config.json` to add your OpenAI API key.

3. Make the run script executable:
   ```
   chmod +x run.sh
   ```

### Usage

1. Place your EPUB files in the `books` directory.

2. Run the translator using the provided script:
   ```
   ./run.sh --book_path /books/your-book.epub --target_lang Spanish
   ```

3. All parameters work the same as in the normal usage, except for the book path which should be prefixed with `/books/` since that's where the Docker container mounts your books directory.

4. The translated EPUB will be saved back to your `books` directory.

### Alternative: Manual Docker Commands

If you prefer to use Docker commands directly:

```bash
# Build the Docker image
docker build -t epub-translator .

# Run with Docker
docker run --rm -v $(pwd)/books:/books -v $(pwd)/config.json:/app/config.json epub-translator --book_path /books/your-book.epub --target_lang Spanish
```

### Using Environment Variables

You can set your OpenAI API key as an environment variable instead of in the config file:

```bash
OPENAI_API_KEY=your-api-key-here ./run.sh --book_path /books/your-book.epub
```
