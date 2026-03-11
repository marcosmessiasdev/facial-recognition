# Auditoria Técnica — Projeto FacialRecognition

## Objetivo da Auditoria
Esta auditoria tem como finalidade validar se a implementação atual do projeto atende aos seguintes critérios:
- Arquitetura implementada conforme planejado
- Pipeline de vídeo funcional
- Integração real com bibliotecas de visão computacional
- Execução de modelos ONNX reais
- Ausência de mocks, stubs ou implementações simuladas
- Aplicação funcional em tempo real
- Overlay visível e sincronizado com a janela analisada

A auditoria também identifica lacunas técnicas, riscos e ajustes necessários antes de considerar o sistema utilizável.

## 1. Verificação da Arquitetura da Solução
**O que verificar:** Confirmar se a solução contém os módulos definidos no planejamento (App, WindowCapture, FramePipeline, FaceDetection, FaceTracking, FaceRecognition, EmotionAnalysis, FaceAttributes, AudioProcessing, IdentityStore, OverlayRenderer).
**Por que isso é necessário:** A separação de módulos garante baixo acoplamento, facilidade de evolução, manutenção mais segura e isolamento de responsabilidades.
**O que não deve acontecer:**
❌ lógica central implementada apenas dentro do projeto WPF
❌ mistura de UI com lógica de visão computacional
❌ dependências cruzadas entre módulos

## 2. Auditoria do Sistema de Captura de Vídeo
**O que verificar:** Confirmar que o módulo WindowCapture utiliza `Windows Graphics Capture API` ou `DirectX Desktop Duplication` (Vortice.Windows).
**Teste funcional obrigatório:** Executar a aplicação e verificar se a janela do Teams pode ser selecionada, frames são capturados continuamente, a captura ocorre em tempo real.
**O que não deve ser feito:**
❌ usar PrintScreen
❌ usar captura GDI (BitBlt)
❌ usar screenshot periódico

## 3. Auditoria do Pipeline de Frames
**O que verificar:** Confirmar que existe um pipeline real de processamento de frames (Capture Thread -> Frame Buffer -> Vision Processing -> Overlay Rendering). Fila de frames, processamento assíncrono e controle de taxa de processamento.
**O que não deve ser feito:**
❌ processar tudo na thread da UI
❌ executar visão computacional diretamente no render da interface

## 4. Auditoria da Detecção de Rostos
**O que verificar:** Confirmar que o módulo FaceDetection executa um modelo ONNX real (ex: `scrfd.onnx`).
**Teste funcional:** Caixas aparecem ao redor dos rostos, múltiplos rostos são detectados, detecção responde ao movimento.
**O que não deve ser feito:**
❌ usar Haar Cascade antigo
❌ retornar dados simulados
❌ detectar apenas um rosto fixo

## 5. Auditoria do Rastreamento Facial
**O que verificar:** Confirmar que existe um módulo de tracking (Ex: KCFTracker, SORT, DeepSORT).
**Teste funcional:** A caixa acompanha o rosto sem piscar constantemente.

## 6. Auditoria do Reconhecimento Facial
**O que verificar:** Confirmar uso de modelo ArcFace ONNX, gerando embeddings de 512 dimensões.
**Teste funcional:** O sistema deve identificar automaticamente pessoas cadastradas que entram/saem da câmera.

## 7. Auditoria da Análise de Emoções
**O que verificar:** Confirmar uso de um modelo ONNX de emoções (ex.: FER+ / FER2013) e que a inferência roda localmente.

## 8. Auditoria da Predição de Gênero (aparência)
**O que verificar:** Confirmar uso de um modelo ONNX de classificação de gênero por aparência (Male/Female), aplicado ao recorte da face, com saída probabilística.
**Observação:** isso não determina gênero biológico, apenas uma classificação visual baseada em aparência.

## 8.1 Auditoria de Atributos (Gênero + Idade em inferência única)
**O que verificar:** Confirmar que, quando disponível, o modelo `genderage.onnx` é carregado e substitui os classificadores separados, reduzindo inferências por face.
**Teste funcional:** Labels de gênero e idade aparecem no overlay sem aumentar a latência perceptivelmente.

## 8.2 Auditoria de Áudio (VAD local)
**O que verificar:** Confirmar captura de áudio local (preferencialmente loopback do sistema) e inferência ONNX do Silero VAD para detectar presença de fala.
**Teste funcional:** Logs registram probabilidade de fala (VAD) e não há travamentos ao iniciar/parar a captura.
**Observação:** O VAD não identifica *quem* fala; ele apenas indica se há fala no áudio.

## 9. Auditoria do Banco de Dados
**O que verificar:** Confirmar uso de SQLite e EF Core. Persistência de embeddings faciais associados a nomes.

## 10. Auditoria do Overlay
**O que verificar:** Confirmar criação de janela transparente, sem borda, sempre no topo e que não bloqueia interação. Overlay acompanha a janela.

## 11. Verificação de Mocks e Implementações Simuladas
**O que verificar:** NENHUM Mock, Fake, Stub, Dummy ou TestData. Todos os módulos devem executar componentes REAIS de inferência e captura.

## 12. Teste de Performance
**Valores aceitáveis:**
- 25–60 FPS
- CPU < 50%
- latência < 100ms

## 13. Verificação de Uso Local
Nenhuma API externa é chamada, sem dependências cloud, sem envio de frames para a internet.

## Critérios de Aprovação Finais
✔ captura da janela funcionar
✔ rostos forem detectados em tempo real
✔ overlay funcionar corretamente
✔ reconhecimento facial funcionar
✔ emoção for analisada
✔ banco persistir dados
✔ nenhum mock estiver presente
