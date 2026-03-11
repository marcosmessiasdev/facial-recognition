# Facial Recognition & Meeting Analytics

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![.NET](https://img.shields.io/badge/.NET-8.0-purple.svg)
![Platform](https://img.shields.io/badge/platform-Windows-0078d7.svg)

Um assistente inteligente em tempo real para análise de reuniões virtuais, focado em privacidade e processamento local.

## 📌 Visão Geral

Este projeto é uma aplicação desktop (.NET 8 / WPF) projetada para observar e analisar reuniões de vídeo (como Microsoft Teams ou Zoom) através da captura de janela. O sistema detecta participantes, reconhece identidades cadastradas, analisa emoções e identifica quem está falando, exibindo tudo em um overlay transparente e não intrusivo.

Diferente de soluções baseadas em nuvem, **todo o processamento ocorre localmente** no computador do usuário, garantindo máxima privacidade e baixa latência.

## ✨ Principais Funcionalidades

- **Detecção Facial de Alta Precisão**: Utiliza o modelo SCRFD para encontrar rostos em tempo real com baixo custo computacional.
- **Rastreamento Temporal (Tracking)**: Mantém a identidade visual dos participantes entre frames, permitindo análises contínuas sem redundância.
- **Reconhecimento de Identidade**: Identifica participantes previamente cadastrados comparando assinaturas faciais (ArcFace) em um banco de dados local (SQLite).
- **Análise de Expressões**: Classifica o estado emocional dos participantes (Feliz, Neutro, Surpreso, etc.).
- **Detecção de Orador Ativo (Active Speaker)**: Combina detecção de atividade de voz (Silero VAD) e análise de movimento da boca via FaceMesh para identificar quem está com a palavra.
- **Transcrição e Analytics**: Gera transcrições automáticas (Whisper) e relatórios de participação, incluindo tempo de fala e interrupções.
- **Interface Overlay**: Renderização de caixas delimitadoras e metadados diretamente sobre a janela da reunião.

## 🛠️ Stack Tecnológica

- **Linguagem/Framework**: C# / .NET 8 / WPF (MVVM)
- **Visão Computacional**: OpenCV (OpenCvSharp4), FaceAiSharp (SCRFD)
- **Machine Learning**: ONNX Runtime (CPU/GPU)
- **Áudio**: NAudio (WASAPI Loopback)
- **Modelos de IA**:
  - SCRFD (Detecção)
  - ArcFace (Reconhecimento)
  - FER2013 (Emoções)
  - Silero VAD (Voz)
  - MediaPipe FaceMesh (Marcos faciais densos)
  - Whisper (Transcrição)
- **Persistência**: Entity Framework Core + SQLite

## 📂 Documentação Detalhada

O projeto possui uma documentação extensa dividida por áreas de interesse:

### 🏛️ Arquitetura e Design
- [Arquitetura do Sistema](docs/architecture.md): Visão geral das camadas e fluxo de dados.
- [Modelo de Domínio](docs/domain-model.md): Entidades principais (Tracks, Person, MeetingSession).
- [Fluxo de Dados](docs/data-flow.md): Detalhamento do pipeline visual e de áudio.

### 🧩 Módulos do Sistema
- [Visão Geral de Módulos](docs/modules.md): Lista completa de componentes.
- [Vision Engine](docs/modules/VisionEngine.md): O cérebro que coordena todo o pipeline.
- [Face Detection](docs/modules/FaceDetection.md) & [Recognition](docs/modules/FaceRecognition.md): Processamento visual profundo.
- [Speaker Detection](docs/modules/SpeakerDetection.md): Fusão de áudio e vídeo para identificar o orador.
- [Audio Processing](docs/modules/AudioProcessing.md): Captura WASAPI e detecção de voz (VAD).
- [Meeting Analytics](docs/modules/MeetingAnalytics.md): Geração de relatórios e métricas de sessão.
- [Window Capture](docs/modules/WindowCapture.md): Captura de alto desempenho via Windows Graphics Capture.

### ⚖️ Decisões Arquiteturais (ADRs)
- [ADR-001: Uso do ONNX Runtime](docs/decisions/ADR-001-onnx-for-inference.md)
- [ADR-002: Estratégia de Matching de Identidade](docs/decisions/ADR-002-identity-matching-strategy.md)

## 🚀 Como Começar

### Pré-requisitos
- **Windows 10 1903 (Build 18362)** ou superior.
- **.NET 8 SDK**.
- Placa de vídeo compatível com DirectX 11 (para captura acelerada).

### Instalação
1. Clone o repositório. O projeto utiliza **Git LFS** para gerenciar os modelos de IA:
   ```bash
   git clone git@github.com:marcosmessiasdev/facial-recognition.git
   cd facial-recognition
   git lfs pull
   ```
2. Compile o projeto:
   ```bash
   dotnet build App/App.csproj
   ```
3. (Opcional) Valide os modelos offline (sem rede):
   ```bash
   pwsh scripts/fetch_models.ps1 -ValidateOnly
   ```
4. Execute a aplicação:
   ```bash
   dotnet run --project App
   ```

### Demo offline (sem YouTube)
- No app, use `🧪 Abrir Demo Offline` e depois selecione a janela do navegador na lista e clique em `▶ Iniciar Análise`.

### E2E offline (determinístico)
1. Instale o Chromium do Playwright (primeira vez):
   ```bash
   pwsh E2ETests/bin/Debug/net8.0-windows10.0.19041.0/playwright.ps1 install chromium
   ```
2. Rode:
   ```bash
   dotnet test E2ETests/E2ETests.csproj --filter "FullyQualifiedName~Test_PipelineStabilityOver30Seconds"
   ```
3. (Opcional) Se quiser GIFs reais para “boca mexendo”, defina `FACIAL_E2E_GIF_DIR` apontando para uma pasta local com `.gif`.

## 🔒 Privacidade e Ética

Este software foi desenvolvido com a **privacidade como prioridade**:
- **Zero Cloud**: Nenhuma imagem ou dado de áudio sai do seu computador.
- **Sem Gravação**: O sistema analisa frames "vivos" e os descarta imediatamente.
- **Conformidade**: Recomendamos que todos os participantes da reunião sejam informados sobre o uso desta ferramenta experimental.

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
