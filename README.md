# Assistente Virtual de Documentos com IA 👩🏻‍💻
Aplicação web para processamento e consulta inteligente de documentos PDF utilizando técnicas de NLP e armazenamento vetorial, o usuário terá a opção de anexar os seus arquivos que serão armazenados no banco de dados e através disso poderá perguntar ao Assistente Virtual (fazer resumos, tirar dúvidas) sobre aquele material que ele anexou.

##  Público Alvo 🎯
**Público Técnicoe/Corporativo:**
- Equipes de TI e desenvolvimento que necessitam integrar processamento documental em fluxos de trabalho
- Startups de LegalTech/RegTech para análise automatizada de contratos e regulamentos
**Setor Educacional:**
- Instituições de ensino para análise de produções acadêmicas
- Pesquisadores que trabalham com grandes volumes de artigos científicos
**Área Jurídica:**
- Escritórios de advocacia para consulta rápida em processos judiciais digitalizados
- Departamentos jurídicos corporativos analisando cláusulas contratuais
**Área Agrícola:**
- Análise de relatórios técnicos de solo e cultivos
- Consulta rápida a normas de segurança agrícola
**Engenharias:**
- Gerenciamento de manuais técnicos de equipamentos
- Análise de normas técnicas (ABNT, ISO)
- Revisão de documentação de obras e licitações

## Funcionalidades Principais ⚙️
- **Upload múltiplo de PDFs** para criação de base de conhecimento
- **Processamento automático** de documentos com divisão inteligente de conteúdo
- **Armazenamento vetorial** usando ChromaDB e embeddings da Hugging Face
- **Interface conversacional** com modelo LLM via Ollama
- **Histórico completo** das interações
- **Persistência de dados** entre sessões

## Pré-requisitos 📝
- Python 3.10+
- Ollama instalado e configurado localmente
  
## Dependências Principais ⚒️
langchain & langchain_chroma: Processamento documental e gestão de cadeias
huggingface-hub: Modelos de embeddings
streamlit: Interface web
ollama: Integração com modelos LLM locais

## Conclusão 📌
![image](https://github.com/user-attachments/assets/1e0531dc-89e8-405e-a264-dfef82a7b431)
