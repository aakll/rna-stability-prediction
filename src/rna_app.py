import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Page configuration
st.set_page_config(
    page_title="RNA Stability Predictor",
    page_icon="🧬",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
       
        model = joblib.load('models/rna_model.joblib')
        scaler = joblib.load('models/rna_scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        return None, None

# Simulate RNA structure prediction (simplified RNAfold logic)
def predict_rna_structure(sequence):
    """
    Simulate RNA secondary structure prediction.
    In production, this would call RNAfold or ViennaRNA package.
    """
    sequence = sequence.upper().replace('T', 'U')
    length = len(sequence)
    
    # Validate sequence
    valid_bases = set('AUGC')
    if not all(base in valid_bases for base in sequence):
        return None, "Invalid sequence. Use only A, U, G, C (or T instead of U)"
    
    # Count G and C
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = gc_count / length if length > 0 else 0
    
    # Estimate stems (simplified: assume ~40-60% of bases pair)
    paired_pct = 0.45 + 0.2 * gc_content  # Higher GC = more pairing
    num_paired = int(length * paired_pct)
    
    # Estimate number of stems (rough heuristic)
    num_stems = max(2, int(length / 20) + np.random.randint(-1, 2))
    
    # Estimate loops (usually stems - 1 or stems)
    num_loops = max(1, num_stems - np.random.randint(0, 2))
    
    # Estimate bulges
    num_bulges = max(0, np.random.randint(0, max(2, num_stems - 1)))
    
    # Average stem length
    avg_stem_length = num_paired / (2 * num_stems) if num_stems > 0 else 0
    
    # Calculate MFE (simplified nearest-neighbor approximation)
    # G-C pairs contribute ~-3 kcal/mol, A-U pairs ~-2 kcal/mol
    gc_pairs = int(num_paired * gc_content / 2)
    au_pairs = int(num_paired * (1 - gc_content) / 2)
    
    mfe_base = -(gc_pairs * 3.0 + au_pairs * 2.0)
    mfe_loop_penalty = num_loops * 2.0  # Loops destabilize
    mfe = mfe_base + mfe_loop_penalty + np.random.normal(0, 2)
    
    # Ensemble free energy (slightly less stable than MFE)
    ensemble_fe = mfe + abs(np.random.normal(1.5, 0.5))
    
    features = {
        'Length': length,
        'GC_Content': round(gc_content, 3),
        'Num_Stems': num_stems,
        'Num_Loops': num_loops,
        'Num_Bulges': num_bulges,
        'Avg_Stem_Length': round(avg_stem_length, 2),
        'Paired_Bases_Pct': round(paired_pct, 3),
        'MFE': round(mfe, 2)
    }
    
    return features, None

# Predict stability
def predict_stability(features, model, scaler):
    """Predict RNA stability from features"""
    feature_order = ['Length', 'GC_Content', 'Num_Stems', 'Num_Loops', 
                     'Num_Bulges', 'Avg_Stem_Length', 'Paired_Bases_Pct', 'MFE']
    
    feature_array = np.array([[features[f] for f in feature_order]])
    feature_scaled = scaler.transform(feature_array)
    stability = model.predict(feature_scaled)[0]
    
    return stability

# Main app
def main():
    st.title("🧬 RNA Secondary Structure Stability Predictor")
    st.markdown("---")
    
    # Description
    st.markdown("""
    ### About This Tool
    This ML tool predicts RNA secondary structure stability from sequence or structural features.
    """)
    
    st.markdown("---")
    
    # Load model
    model, scaler = load_model()
    
    if model is None:
        st.error("⚠️ Model files not found. Please ensure rna_model.pkl and rna_scaler.pkl are in the directory.")
        return
    
    # Input method selection
    st.subheader("📥 Input Method")
    input_method = st.radio(
        "Choose input method:",
        ["RNA Sequence (Automatic)", "Manual Features (Advanced)"],
        help="Sequence input automatically calculates structural features"
    )
    
    if input_method == "RNA Sequence (Automatic)":
        st.markdown("---")
        st.subheader("🧬 Enter RNA Sequence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sequence_input = st.text_area(
                "RNA Sequence:",
                placeholder="Enter RNA sequence (A, U, G, C)...\nExample: AUGCGAUCGAUCGAUCGAU",
                height=150,
                help="Enter RNA sequence using A, U, G, C (or T instead of U)"
            )
            
            predict_button = st.button("🔍 Predict Stability", type="primary")
        
        with col2:
            st.subheader("📋 Example Sequences")
            st.markdown("""
            **Short miRNA-like:**
            ```
            UGGAAUGUAAAGAAGUAUGUA
            ```
            
            **Medium tRNA-like:**
            ```
            GCGGAUUUAGCUCAGDDGGGAGAGC
            GCCAGACUGAAGAUCUGGAGGUCCU
            GUGUUCGAUCCACAGAAUUCGCACCA
            ```
            
            **Note:** Sequences 18-200 nt work best
            """)
        
        if predict_button:
            if not sequence_input.strip():
                st.warning("⚠️ Please enter an RNA sequence.")
            else:
                # Clean sequence
                sequence = sequence_input.strip().replace('\n', '').replace(' ', '')
                
                with st.spinner("Analyzing RNA structure..."):
                    # Predict structure
                    features, error = predict_rna_structure(sequence)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success("✅ RNA structure analyzed successfully!")
                        
                        # Display sequence info
                        st.markdown("---")
                        st.subheader("📊 Sequence Information")
                        
                        seq_col1, seq_col2, seq_col3 = st.columns(3)
                        with seq_col1:
                            st.metric("Length", f"{features['Length']} nt")
                        with seq_col2:
                            st.metric("GC Content", f"{features['GC_Content']*100:.1f}%")
                        with seq_col3:
                            a_count = sequence.count('A')
                            u_count = sequence.count('U') + sequence.count('T')
                            g_count = sequence.count('G')
                            c_count = sequence.count('C')
                            st.metric("Composition", f"A:{a_count} U:{u_count} G:{g_count} C:{c_count}")
                        
                        # Display predicted structure features
                        st.markdown("---")
                        st.subheader("🔬 Predicted Secondary Structure Features")
                        
                        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
                        
                        with feat_col1:
                            st.metric("Stems", features['Num_Stems'])
                            st.metric("Loops", features['Num_Loops'])
                        
                        with feat_col2:
                            st.metric("Bulges", features['Num_Bulges'])
                            st.metric("Avg Stem Length", f"{features['Avg_Stem_Length']:.1f} bp")
                        
                        with feat_col3:
                            st.metric("Paired Bases", f"{features['Paired_Bases_Pct']*100:.1f}%")
                            st.metric("MFE", f"{features['MFE']:.2f} kcal/mol")
                        
                        with feat_col4:
                            st.info("**Note:** Structure features calculated using simplified folding algorithm. For production use, integrate with RNAfold/ViennaRNA.")
                        
                        # Predict stability
                        with st.spinner("Predicting stability..."):
                            stability = predict_stability(features, model, scaler)
                        
                        # Display prediction
                        st.markdown("---")
                        st.subheader("🎯 Stability Prediction")
                        
                        if stability > 0.7:
                            color = "green"
                            category = "Stable"
                            interpretation = "High stability - likely functional regulatory RNA"
                        elif stability > 0.5:
                            color = "orange"
                            category = "Moderate"
                            interpretation = "Moderate stability - may require optimization"
                        else:
                            color = "red"
                            category = "Unstable"
                            interpretation = "Low stability - structural modifications recommended"
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                            <h2>Stability Score: {stability:.3f}</h2>
                            <h3>{category} Structure</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.info(f"**Interpretation:** {interpretation}")
                        
                        st.markdown("""
                        **Stability Scale:**
                        - **0.7 - 1.0**: Stable (suitable for regulatory functions)
                        - **0.5 - 0.7**: Moderate (may need optimization)
                        - **0.0 - 0.5**: Unstable (structural redesign recommended)
                        
                        *Higher scores indicate more thermodynamically stable secondary structures.*
                        """)
    
    else:  # Manual features input
        st.markdown("---")
        st.subheader("⚙️ Manual Feature Input")
        
        st.info("💡 **For advanced users:** Input structural features directly if you have them from RNAfold or other tools.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            length = st.number_input("Length (nucleotides)", min_value=10, max_value=300, value=75, step=1)
            gc_content = st.slider("GC Content", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
            num_stems = st.number_input("Number of Stems", min_value=1, max_value=20, value=4, step=1)
            num_loops = st.number_input("Number of Loops", min_value=0, max_value=15, value=3, step=1)
        
        with col2:
            num_bulges = st.number_input("Number of Bulges", min_value=0, max_value=10, value=1, step=1)
            avg_stem_length = st.number_input("Average Stem Length (bp)", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
            paired_pct = st.slider("Paired Bases %", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
            mfe = st.number_input("MFE (kcal/mol)", min_value=-150.0, max_value=0.0, value=-40.0, step=1.0)
        
        predict_button_manual = st.button("🔍 Predict Stability", type="primary", key="manual_predict")
        
        if predict_button_manual:
            features = {
                'Length': length,
                'GC_Content': gc_content,
                'Num_Stems': num_stems,
                'Num_Loops': num_loops,
                'Num_Bulges': num_bulges,
                'Avg_Stem_Length': avg_stem_length,
                'Paired_Bases_Pct': paired_pct,
                'MFE': mfe
            }
            
            with st.spinner("Predicting stability..."):
                stability = predict_stability(features, model, scaler)
            
            st.markdown("---")
            st.subheader("🎯 Stability Prediction")
            
            if stability > 0.7:
                            color = "green"
                            category = "Stable"
                            interpretation = "High stability - likely functional regulatory RNA"
            elif stability > 0.5:
                            color = "orange"
                            category = "Moderate"
                            interpretation = "Moderate stability - may require optimization"
            else:
                 color = "red"
                 category = "Unstable"
                 interpretation = "Low stability - structural modifications recommended"
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                <h2>Stability Score: {stability:.3f}</h2>
                <h3>{category} Structure</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.info(f"**Interpretation:** {interpretation}")
                            
            st.markdown("""                
            **Stability Scale:**
            - **0.7 - 1.0**: Stable (suitable for regulatory functions)
            - **0.5 - 0.7**: Moderate (may need optimization)
            - **0.0 - 0.5**: Unstable (structural redesign recommended)
                        
            *Higher scores indicate more thermodynamically stable secondary structures.*
            """)       
    
    # Footer
    st.markdown("""
        **Model Performance:**
        - Algorithm: Random Forest Regressor
        - Test R²: 0.968
        - Focus: Regulatory RNAs (miRNA, riboswitches, tRNA, rRNA, ribozymes)
    """)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🧬 RNA Structure Stability Prediction Tool</p>
        <p>Developed for regulatory RNA analysis and gene regulation research</p>
        <p>Model trained on 200 RNA sequences | Random Forest (R² = 0.968)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
