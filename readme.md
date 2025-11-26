# Check-in de Hábitos Saudáveis com Reconhecimento Facial

Este projeto implementa um sistema local de detecção e reconhecimento facial usando OpenCV, Haar Cascade e LBPH para registrar um check-in de hábito saudável como beber água, alongar ou meditar.  
A aplicação roda totalmente no desktop ou notebook, sem internet, servidor ou app externo.

# Objetivo

Criar uma aplicação simples que detecta e reconhece o usuário pela câmera e registra um check-in de um hábito saudável.  
O projeto atende os requisitos da disciplina: uso de IA/ML, parâmetros ajustáveis, exibição da detecção e código executável.

# Tecnologias Utilizadas

- Python 3
- OpenCV (opencv-contrib-python)
- Haar Cascade
- LBPH Face Recognizer
- NumPy

# Estrutura do Projeto

```
health_checkin/
├── setup_user.py
├── habit_checkin.py
├── user_name.txt
├── face_model.yml
├── haarcascade_frontalface_default.xml
├── checkins.csv
└── requirements.txt
```

# Como Executar

## 1. Instalar dependências

```
pip install -r requirements.txt
```

## 2. Cadastrar o rosto e treinar o modelo

```
python setup_user.py
```

Digite seu nome.  
Olhe para a câmera até a captura chegar a 30 imagens.  
O modelo será treinado e salvo automaticamente.

## 3. Rodar o check-in

```
python habit_checkin.py
```

Digite o hábito.  
Olhe para a câmera.  
Seu nome aparecerá em verde quando for reconhecido.  
Pressione `c` para registrar o check-in.  
Pressione `q` para sair.

Os registros ficam em:

```
checkins.csv
```

# Parâmetros Ajustáveis e Impacto

O projeto demonstra parâmetros relevantes tanto da detecção facial quanto do reconhecimento.

## Parâmetros da Detecção (Haar Cascade)

Estes parâmetros afetam o retângulo que detecta o rosto:

| Parâmetro | Aumentar | Diminuir |
|----------|----------|-----------|
| SCALE_FACTOR | Mais rápido, pode perder rosto | Mais preciso, mais lento |
| MIN_NEIGHBORS | Menos falsos positivos, pode falhar | Mais detecções, pode errar |
| MIN_SIZE | Evita rostos pequenos | Detecta mais, pode ficar instável |

## Parâmetros do Reconhecimento (LBPH)

| Parâmetro | Função |
|----------|--------|
| CONFIDENCE_LIMIT | Exigência do reconhecimento |
| radius / neighbors / grid_x / grid_y | Sensibilidade do modelo |

Exemplos para demonstrar no vídeo:  
- CONFIDENCE_LIMIT = 40 → reconhecimento mais exigente  
- CONFIDENCE_LIMIT = 90 → reconhecimento mais permissivo  

# Nota Ética sobre Dados Faciais

Este projeto é apenas educacional e roda localmente.  
Dados faciais são sensíveis e devem ser tratados com cuidado.

Recomendações:
- Não subir imagens do rosto para repositórios públicos  
- Não compartilhar o arquivo face_model.yml  
- Remover dados após o uso  
- Em aplicações reais: política de privacidade, LGPD, consentimento e direito de exclusão

# Dependências

```
opencv-contrib-python
numpy
```
