import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import plotly.express as px

# Configuração da Página

st.set_page_config(
    page_title="Passos Mágicos - Predição de Risco",
    page_icon="🎓",
    layout="wide"
)

# Carregar Modelo e Encoders

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('modelo/modelo_xgboost.pkl')
        encoders = joblib.load('modelo/encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        st.error("Arquivos de modelo não encontrados! Verifique a pasta 'modelo'.")
        return None, None

model, encoders = load_assets()

# Título e Sidebar

st.title("🎓 Sistema de Previsão de Risco - Passos Mágicos")
st.sidebar.header("Navegação")

# Navegação

page = st.sidebar.radio("Escolha o Modo:", ["🔮 Simulador Individual", "📊 Dashboard & Upload"])

# PÁGINA: SIMULADOR INDIVIDUAL

if page == "🔮 Simulador Individual":
    st.markdown("### Simulador de Risco Individual")
    st.write("Preencha os dados de um aluno hipotético para verificar a probabilidade de risco.")

    if model is not None:
        with st.form("simulador_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                FASE = st.selectbox("FASE", options=encoders['FASE'].classes_)
                
                # Visualmente apenas Feminino/Masculino
                
                GENERO = st.selectbox("Gênero", options=["Feminino", "Masculino"])
                
                # FILTRO DAS OPÇÕES DE PEDRA
                
                opcoes_pedra = [p for p in encoders['PEDRA'].classes_ if p not in ['nan']]
                PEDRA = st.selectbox("PEDRA", options=opcoes_pedra)

            with col2:
                inde = st.number_input("INDE (Índice Geral)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
                ida = st.number_input("IDA (Desempenho Acadêmico)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
                ieg = st.number_input("IEG (Engajamento)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)

            with col3:
                iaa = st.number_input("IAA (Autoavaliação)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
                ips = st.number_input("IPS (Psicossocial)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
                DEFASAGEM = st.number_input("Nível de DEFASAGEM (Anos)", min_value=-10, max_value=5, value=0, step=1, help="0 = Série correta. -1 = Atrasado 1 ano.")

            submit_button = st.form_submit_button(label="Calcular Risco")

        if submit_button:
            input_data = pd.DataFrame({
                'FASE': [FASE],
                'GENERO': [GENERO],
                'INDE': [inde],
                'PEDRA': [PEDRA],
                'IAA': [iaa],
                'IEG': [ieg],
                'IPS': [ips],
                'IDA': [ida],
                'DEFASAGEM': [DEFASAGEM]
            })

            # Tratamento de Encoders
            
            input_data['FASE'] = encoders['FASE'].transform(input_data['FASE'])
            input_data['PEDRA'] = encoders['PEDRA'].transform(input_data['PEDRA'])
            
            # GENERO
            
            try:
                input_data['GENERO'] = encoders['GENERO'].transform(input_data['GENERO'])
            except Exception as e:
                st.error("Erro no encoding de gênero. Certifique-se de que o modelo reconhece as opções 'Feminino' e 'Masculino'.")

            # Predição
            
            prob = model.predict_proba(input_data)[0][1]
            risco = prob > 0.5 

            # Resultado Visual
            
            st.markdown("---")
            st.subheader("Resultado da Simulação")
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                if risco:
                    st.error(f"⚠️ **ALTO RISCO DETECTADO**")
                    st.metric(label="Probabilidade", value=f"{float(prob):.1%}")
                else:
                    st.success(f"✅ **BAIXO RISCO**")
                    st.metric(label="Probabilidade", value=f"{float(prob):.1%}")

            with c2:
                st.write("Termômetro de Risco:")
                st.progress(float(prob))
                if risco:
                    st.warning("Recomendação: Iniciar intervenção pedagógica focada em melhoria do IDA e IEG.")
                else:
                    st.info("Recomendação: Monitorar manutenção dos índices atuais.")

# PÁGINA: DASHBOARD & UPLOAD

elif page == "📊 Dashboard & Upload":
    st.markdown("### Análise em Massa de Alunos")
    st.info("Faça o upload da planilha atual (CSV ou Excel) para identificar alunos em risco.")

    uploaded_file = st.file_uploader("Carregar arquivo de dados", type=['csv', 'xlsx'])

    if uploaded_file and model is not None:
        try:
            
            # Leitura Inteligente do Arquivo
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                if len(sheet_names) > 1:
                    st.success(f"Arquivo Excel com {len(sheet_names)} abas encontradas.")
                    selected_sheet = st.selectbox("📂 Selecione a aba para analisar:", sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    df = pd.read_excel(uploaded_file)

            # Padronização de Colunas
            
            mapa_flexivel = {
                'NOME': ['Nome', 'Nome Anonimizado', 'Nome Aluno'],
                'FASE': ['Fase', 'FASE'],
                'GENERO': ['Gênero', 'Genero', 'GENERO'],
                'INDE': ['INDE', 'INDE 2025','INDE 2024', 'INDE 2023', 'INDE 22', 'INDE 2022'],
                'PEDRA': ['Pedra', 'Pedra 2025',  'Pedra 2024', 'Pedra 2023', 'Pedra 22', 'PEDRA'],
                'IAA': ['IAA'],
                'IEG': ['IEG'],
                'IPS': ['IPS'],
                'IDA': ['IDA'],
                'DEFASAGEM': ['Defasagem', 'Defas', 'DEFASAGEM']
            }

            for padrao, variacoes in mapa_flexivel.items():
                for var in variacoes:
                    if var in df.columns:
                        df = df.rename(columns={var: padrao})
                        break 

            # CORREÇÃO DE TIPOS DE DADOS
            
            # Converte colunas numéricas que podem estar como texto (ex: "7,5")
            
            cols_numericas = ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'DEFASAGEM']
            
            for col in cols_numericas:
                if col in df.columns:
                    
                    # Substitui vírgula por ponto e converte para numérico
                    
                    df[col] = df[col].astype(str).str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Validar Colunas
            
            features_model = ['FASE', 'GENERO', 'INDE', 'PEDRA', 'IAA', 'IEG', 'IPS', 'IDA', 'DEFASAGEM']
            missing_cols = [col for col in features_model if col not in df.columns]

            if missing_cols:
                st.error(f"⚠️ A aba selecionada não possui as colunas necessárias. Faltando: {missing_cols}")
                st.write("Colunas encontradas:", list(df.columns))
            else:
                X_input = df[features_model].copy().dropna()
                
                # Encoders
                
                for col in ['FASE', 'PEDRA', 'GENERO']:
                    X_input[col] = X_input[col].astype(str).apply(
                        lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
                    )
                
                X_input = X_input[X_input['FASE'] != -1]

                if len(X_input) == 0:
                    st.warning("Nenhum dado válido para processar após a limpeza. Verifique se as colunas numéricas (INDE, IDA...) contêm números.")
                else:
                    
                    # Predição
                    
                    probs = model.predict_proba(X_input)[:, 1]
                    preds = model.predict(X_input)

                    df_results = df.loc[X_input.index].copy()
                    df_results['Risco_Predito'] = preds
                    df_results['PROBABILIDADE_RISCO'] = probs

                    # EXIBIÇÃO
                    
                    total_alunos = len(df_results)
                    total_risco = df_results['Risco_Predito'].sum()
                    perc_risco = (total_risco / total_alunos) * 100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total de Alunos Analisados", total_alunos)
                    col2.metric("Alunos em Risco (Alerta)", int(total_risco), delta_color="inverse")
                    col3.metric("% da Turma em Risco", f"{perc_risco:.1f}%")

                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    
                    # Gráfico 1: Risco por Fase
                    
                    risk_by_phase = df_results[df_results['Risco_Predito'] == 1].groupby('FASE').size().reset_index(name='Contagem')
                    
                    if not risk_by_phase.empty:
                        fig_bar = px.bar(
                            risk_by_phase, 
                            x='FASE', 
                            y='Contagem', 
                            title="Alunos em Risco por Fase", 
                            color='Contagem', 
                            color_continuous_scale='Reds',
                            text='Contagem', 
                            labels={'FASE': 'Fase', 'Contagem': 'Quantidade de Alunos'}
                        )
                        fig_bar.update_traces(textposition='outside')
                        c1.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        c1.info("Parabéns! Nenhum aluno em risco detectado.")

                    # Gráfico 2: Distribuição de Risco 
                    
                    fig_hist = px.histogram(
                        df_results, 
                        x='PROBABILIDADE_RISCO', 
                        nbins=20, 
                        title="Distribuição de Risco da Turma", 
                        color_discrete_sequence=['#636EFA'],
                        text_auto=True,
                        labels={'PROBABILIDADE_RISCO': 'Probabilidade de Risco (%)', 'count': 'Contagem de Alunos'} 
                    )
                    fig_hist.update_layout(yaxis_title="Contagem de Alunos")
                    fig_hist.update_traces(textposition='outside')
                    
                    c2.plotly_chart(fig_hist, use_container_width=True)

                    # Tabela com as probabilidades
                    
                    st.markdown("### 🚨 Lista de Prioridade")
                    
                    cols_show = ['RA', 'NOME', 'FASE', 'INDE', 'DEFASAGEM', 'PROBABILIDADE_RISCO']
                    cols_show = [c for c in cols_show if c in df_results.columns]

                    top_risk = df_results.sort_values('PROBABILIDADE_RISCO', ascending=False)
                    top_risk = top_risk[cols_show].copy()

                    if 'INDE' in top_risk.columns:
                        top_risk['INDE'] = top_risk['INDE'].round(2)
                    
                    if 'PROBABILIDADE_RISCO' in top_risk.columns:
                        top_risk['PROBABILIDADE_RISCO'] = top_risk['PROBABILIDADE_RISCO'].round(4)

                    st.dataframe(
                        top_risk.style
                        .background_gradient(subset=['PROBABILIDADE_RISCO'], cmap='Reds')
                        .format({
                            'PROBABILIDADE_RISCO': '{:.2%}', 
                            'INDE': '{:.2f}'
                        }),
                        use_container_width=True
                    )

                    csv = top_risk.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar Relatório Completo (CSV)",
                        data=csv,
                        file_name='relatorio_risco_passos_magicos.csv',
                        mime='text/csv',
                    )

        except Exception as e:

            st.error(f"Erro ao processar o arquivo: {e}")
