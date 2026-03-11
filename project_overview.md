# Visão Geral do Projeto

Este projeto tem como objetivo desenvolver um aplicativo desktop experimental capaz de analisar reuniões de vídeo em tempo real (por exemplo, reuniões realizadas no Microsoft Teams) para detectar rostos, identificar participantes previamente cadastrados e analisar expressões faciais, exibindo essas informações diretamente na tela durante a reunião.

A aplicação será executada localmente no computador do usuário, sem enviar dados para a nuvem. Todo o processamento de vídeo e inteligência artificial ocorrerá no próprio dispositivo, utilizando modelos de visão computacional open source. O projeto tem caráter experimental e exploratório, visando estudar técnicas modernas de visão computacional aplicadas a interações humanas em ambientes de reunião virtual.

O objetivo não é modificar ou integrar diretamente com o Teams, mas sim observar a reunião através da captura da janela da aplicação e aplicar algoritmos de análise visual sobre os frames exibidos na tela.

# Objetivos do Projeto

O sistema deverá ser capaz de realizar as seguintes funções:

1. Detecção de rostos em tempo real
Durante a reunião, o sistema capturará continuamente os frames da janela do Teams e identificará automaticamente todos os rostos visíveis na tela.

2. Rastreamento dos rostos
Após detectar os rostos, o sistema manterá o rastreamento contínuo de cada rosto entre os frames, evitando recalcular a detecção completa a cada atualização do vídeo.

3. Identificação de participantes
Caso um participante já tenha sido previamente cadastrado no sistema, o software poderá reconhecer automaticamente o rosto e associar o nome correspondente.
Quando um rosto desconhecido aparecer, o sistema poderá permitir que o usuário associe manualmente um nome ao rosto detectado.

4. Análise de expressões faciais
A aplicação poderá analisar expressões faciais básicas e classificá-las em categorias emocionais aproximadas, como:
- Feliz
- Neutro
- Surpreso
- Triste
- Irritado
Essas informações podem ser exibidas como indicadores visuais durante a reunião.

5. Exibição de informações em overlay
Os resultados da análise serão exibidos em uma camada gráfica sobre a janela do Teams, sem modificar o aplicativo original. Essa camada apresentará informações como:
- Caixa delimitando o rosto detectado
- Nome do participante (quando identificado)
- Emoção estimada

# Arquitetura Geral da Solução

A arquitetura da aplicação será baseada em um pipeline de processamento de vídeo em tempo real.

Fluxo de processamento:
Captura da janela do Teams
        ↓
Extração de frames de vídeo
        ↓
Detecção de rostos
        ↓
Rastreamento dos rostos
        ↓
Reconhecimento facial (opcional)
        ↓
Detecção de emoção
        ↓
Renderização de overlay

Cada etapa será implementada como um módulo independente dentro da aplicação.

# Stack Tecnológica

A solução utilizará exclusivamente tecnologias gratuitas e open source.

## Plataforma da Aplicação
- .NET 9
- WPF para interface desktop
- Arquitetura baseada em MVVM

Motivo da escolha: forte integração com Windows, excelente suporte a rendering, bom suporte a aplicações desktop de alto desempenho.

## Captura da janela do Teams
A captura de vídeo será feita utilizando a API moderna do Windows:
- Windows Graphics Capture API
Para integração com .NET será utilizada a biblioteca:
- Vortice.Windows

Essa tecnologia permite capturar frames diretamente da GPU com alta performance, possibilitando taxas de atualização entre 30 e 60 frames por segundo.

## Processamento de visão computacional
Para manipulação de imagens e suporte a algoritmos de visão computacional será utilizada a biblioteca:
- OpenCvSharp

Essa biblioteca é um wrapper .NET para o OpenCV e fornece recursos para manipulação de frames, transformação de imagens, tracking de objetos.

## Motor de inferência de IA
Para executar os modelos de inteligência artificial será utilizado:
- ONNX Runtime

Motivos da escolha: execução local, compatível com CPU e GPU, integração direta com .NET, alto desempenho.

## Modelos de Inteligência Artificial
Todos os modelos utilizados serão open source.

- Detecção de rosto: SCRFD (detecção rápida e precisa, ideal para aplicações em tempo real)
- Reconhecimento facial: ArcFace (gera um embedding facial, comparado com banco local)
- Detecção de emoções: FER2013 CNN (classifica expressões faciais básicas)

## Armazenamento de dados
Para armazenar informações de identificação facial será utilizado um banco de dados local.
- Tecnologia: SQLite
- Dados armazenados: nome do participante, embedding facial, metadados opcionais.
Todo armazenamento permanecerá no computador do usuário.

## Renderização da Interface
Para desenhar as informações sobre a reunião será utilizada uma janela overlay transparente.
- Tecnologia: WPF Drawing API
Características da overlay: transparente, sempre sobre a janela do Teams, não interfere com cliques do usuário, atualização em tempo real.

## Estrutura de módulos do sistema
A aplicação será organizada em módulos especializados:
App
 ├── WindowCapture
 ├── AudioProcessing
 ├── FramePipeline
 ├── FaceDetection
 ├── FaceTracking
 ├── FaceRecognition
 ├── EmotionAnalysis
 ├── FaceAttributes (Gender+Age)
 ├── GenderAnalysis (fallback)
 ├── AgeAnalysis (fallback)
 ├── IdentityStore
 └── OverlayRenderer

Essa organização permite que cada componente evolua independentemente.

## Privacidade e processamento local
Um princípio importante deste projeto é que todo o processamento ocorre localmente.
O sistema não envia imagens para servidores, não depende de serviços externos, não armazena vídeo da reunião, e utiliza apenas embeddings faciais para identificação.

## Possíveis extensões futuras
- identificação automática de quem está falando (Active Speaker Detection: áudio + movimento da boca)
- geração automática de atas de reunião
- métricas de engajamento dos participantes
- análise de comportamento em apresentações
- dashboards de interação em reuniões

## Active Speaker Detection (implementação atual)
- Captura de áudio por loopback (WASAPI) + Silero VAD (ONNX) para detectar presença de fala.
- “Boca mexendo” estimada por movimento na região da boca (ROI), usando landmarks do SCRFD quando disponíveis.
- Quando `speech=true`, o overlay destaca em amarelo o rosto com maior score e mostra `[Speaking]`.
- Ao parar a pipeline, uma sessão é salva em `logs/meeting_session_*.json` com segmentos e métricas básicas.

## Conclusão
Este projeto propõe a construção de um sistema de análise visual de reuniões em tempo real, combinando captura de vídeo, visão computacional e inteligência artificial local. A aplicação utilizará tecnologias modernas e gratuitas, com foco em alto desempenho, processamento local e modularidade. O resultado será um protótipo funcional capaz de detectar e analisar participantes em reuniões virtuais sem modificar ou integrar diretamente com a plataforma de videoconferência utilizada.

# Diretrizes Adicionais e Boas Práticas

## 1. Escopo do projeto (o que o sistema NÃO fará)
É importante deixar explícito o limite do sistema para evitar expectativas erradas.
O projeto não pretende:
- modificar o Microsoft Teams
- integrar com APIs internas do Teams
- gravar reuniões
- armazenar vídeo ou imagens das pessoas
- enviar dados para servidores externos
- funcionar como ferramenta de vigilância
O sistema será apenas um observador visual local, analisando a janela exibida na tela.

## 2. Uso estritamente local
Uma diretriz importante do projeto é que todo o processamento ocorre localmente.
Isso significa:
- nenhum frame será enviado para a internet
- nenhuma API externa será utilizada
- nenhum serviço cloud será necessário
Benefícios: maior privacidade, menor latência, independência de internet, custo zero de operação.

## 3. Consentimento dos participantes
Como o sistema envolve análise facial, é recomendável adicionar uma política de uso:
- todos os participantes devem estar cientes do experimento
- o uso deve ocorrer apenas em ambiente controlado
- o cadastro facial deve ser opcional
- qualquer participante pode solicitar a remoção do seu perfil
Mesmo sendo um projeto local, essa transparência é importante.

## 4. Limitações técnicas esperadas
A documentação deve explicar que o sistema possui limitações naturais. Exemplo:
- resolução da imagem (se o rosto for pequeno na tela, a detecção pode falhar)
- iluminação ruim
- ângulo do rosto
- mudanças dinâmicas no layout do Teams (grid view, speaker view). O sistema precisa se adaptar.

## 5. Estratégia de performance
Processar vídeo em tempo real exige cuidados. Decisões arquiteturais importantes:
- **pipeline assíncrono**: Separar captura, processamento e renderização em threads diferentes (Capture Thread -> Frame Queue -> Vision Processing Thread -> Overlay Rendering Thread).
- **redução de carga**: detectar rostos a cada N frames e usar tracking nos frames intermediários. Isso reduz o uso de CPU.

## 6. Controle de taxa de processamento
O sistema não precisa rodar a 60 FPS nas operações pesadas. Taxas sugeridas:
- captura: 30 fps
- detecção facial: 10 fps
- tracking: 30 fps
- reconhecimento: sob demanda
- emoção: 5 fps

## 7. Configuração do sistema
Incluir um arquivo de configuração para permitir ajustes sem recompilar. Exemplo:
```json
{
  "faceDetectionInterval": 5,
  "emotionDetectionInterval": 10,
  "trackingEnabled": true,
  "recognitionThreshold": 0.75
}
```

## 8. Registro de logs
Adicionar um sistema de logging (Serilog recomendado) ajudará no desenvolvimento, registrando: número de rostos detectados, tempo de processamento, erros de inferência, eventos de reconhecimento.

## 9. Sistema de calibração inicial
Permitir um modo de calibração onde o usuário pode cadastrar o rosto, confirmar identificação e ajustar thresholds. (Fluxo: capturar rosto -> gerar embedding -> associar nome -> salvar no banco).

## 10. Testes com datasets públicos
Usar datasets conhecidos para validar precisão antes de uso real: FER2013/AffectNet para emoção, LFW para reconhecimento.

## 11. Métricas de qualidade
Adicionar métricas para avaliar o experimento: taxa de detecção facial, taxa de reconhecimento correto, latência de processamento, FPS médio.

## 12. Segurança dos embeddings
Embeddings devem ser tratados com cuidado: armazenar apenas o vetor, não armazenar imagens, e criptografar o banco local (opcional).

## 13. Estrutura de roadmap
- **Fase 1 (MVP)**: captura, detecção de rostos, overlay visual.
- **Fase 2 (Tracking)**: rastreamento de rostos para melhor performance.
- **Fase 3 (Reconhecimento)**: cadastro de pessoas e identificação automática.
- **Fase 4 (Emoções)**: classificação de expressões.

## 14. Possíveis evoluções futuras
- Identificação de quem está falando.
- Transcrição da reunião.
- Análise de engajamento e geração automática de resumo/estatísticas.
