# Comedy-Central

## Overview

Comedy-Central is a project dedicated to transforming information streams into comedic content, delivering smiles through humor. Our mission is to make information engaging and entertaining by presenting it in a lighthearted, comedic format. We believe in the power of smiles to enhance user experiences.

**Project Repository**: [https://github.com/iamorangedad/Comedy-Central](https://github.com/iamorangedad/Comedy-Central)

## Project Modules

The Comedy-Central project is structured into three core modules, each contributing to the goal of delivering comedic content seamlessly:

### Module 1: Web Content to Comedy Script Conversion
- **Purpose**: Transform web content (e.g., articles, text, or data streams) into humorous scripts suitable for comedic delivery.
- **Functionality**:
  - Extracts text from web pages using parsing techniques.
  - Analyzes content to identify key points and context.
  - Generates comedic scripts by injecting humor (e.g., puns, witty remarks, or satirical spins) while preserving the core information.
  - Utilizes natural language processing (NLP) to ensure the tone aligns with comedic styles (e.g., slapstick, dry humor, or stand-up).
- **Technologies**: Python (for NLP), JavaScript (for web scraping), or libraries like BeautifulSoup and NLTK.
- **Output**: A structured comedic script ready for further processing.

### Module 2: Text-to-Speech (TTS) Comedy Audio Generation
- **Purpose**: Convert the comedic scripts generated in Module 1 into audio output with a humorous tone.
- **Functionality**:
  - Processes the comedic script into audio using TTS engines.
  - Applies voice modulation to match comedic delivery (e.g., exaggerated tones, pauses for punchlines).
  - Supports multiple voices or accents to enhance the humor.
  - Ensures audio clarity and compatibility with various playback formats.
- **Technologies**: TTS libraries (e.g., Google Text-to-Speech, Amazon Polly), audio processing tools.
- **Output**: An audio file (e.g., MP3 or WAV) containing the comedic performance of the script.

### Module 3: Browser Extension Integration
- **Purpose**: Combine Modules 1 and 2 into a cohesive browser extension that delivers comedic content directly to users.
- **Functionality**:
  - Integrates with web browsers (e.g., Chrome, Firefox) to process web content in real-time.
  - Automatically extracts content from active web pages (via Module 1).
  - Converts the extracted content into comedic scripts and generates audio output (via Module 2).
  - Provides a user interface (e.g., a popup or sidebar) to trigger the comedic transformation and play audio.
  - Allows customization (e.g., selecting humor styles or voice preferences).
- **Technologies**: JavaScript, HTML, CSS (for the extension), WebExtension APIs, and integration with Modules 1 and 2.
- **Output**: A browser extension that transforms web content into comedic audio on demand.

## Getting Started

### Prerequisites
- Python 3.x (for Module 1 scripts)
- Node.js (for browser extension development)
- TTS service API keys (e.g., Google Cloud, AWS)
- Browser extension development environment (e.g., Chrome Web Store account)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/iamorangedad/Comedy-Central.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Comedy-Central
   ```
3. Install dependencies for Module 1 (example):
   ```bash
   pip install beautifulsoup4 nltk
   ```
4. Set up Module 2 by configuring your TTS service (refer to provider documentation).
5. For Module 3, install Node.js dependencies for the browser extension:
   ```bash
   npm install
   ```

### Usage
1. **Module 1**: Run the script to convert web content to a comedic script:
   ```bash
   python scripts/content_to_script.py --url <webpage-url>
   ```
2. **Module 2**: Generate audio from the script:
   ```bash
   python scripts/script_to_audio.py --script <script-file>
   ```
3. **Module 3**: Load the browser extension in your browser (e.g., Chrome):
   - Go to `chrome://extensions/`, enable "Developer mode," and load the `extension/` folder.
   - Click the extension icon to transform and play comedic content from the current webpage.

## Contributing
We welcome contributions to make Comedy-Central even funnier! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, open an issue on the [GitHub repository](https://github.com/iamorangedad/Comedy-Central) or contact the maintainers.

---

**Comedy-Central**: Because laughter is the best way to learn!