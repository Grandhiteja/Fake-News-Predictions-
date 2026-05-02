import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import os

# --- Page Config ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- Custom CSS for clean UI ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        font-size: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    .real {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Architecture (Must match training code exactly) ---
class TinyBertCrossAttnBiGRU(nn.Module):
    def __init__(self, model_name, lora_r=4, lora_alpha=32, num_heads=8, gru_hidden=64):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # LoRA Config
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['query', 'value'],
            lora_dropout=0.1,
            bias='none',
            task_type='SEQ_CLS'
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * gru_hidden, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, h_ids, h_mask, b_ids, b_mask):
        h = self.backbone(input_ids=h_ids, attention_mask=h_mask).last_hidden_state
        b = self.backbone(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
        
        cross, _ = self.cross_attn(query=h, key=b, value=b, key_padding_mask=(b_mask == 0))
        combined = torch.cat([h, cross], dim=1)
        
        gru_out, _ = self.gru(combined)
        pooled, _ = torch.max(gru_out, dim=1)
        
        x = self.dropout(pooled)
        logit = self.fc(x)
        return self.sigmoid(logit).squeeze(-1)

# --- Load Model Function ---
@st.cache_resource
def load_model():
    model_path = "tinybert_fake_news_model"  # Matches your folder name
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model folder '{model_path}' not found! Please check your directory.")
        st.stop()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer from local folder
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        st.stop()

    # Initialize model structure
    # Note: We need the base model name used during training to initialize structure
    # If internet is available, it will fetch config. If not, it relies on local files.
    base_model_name = 'huawei-noah/TinyBERT_General_4L_312D' 
    
    try:
        model = TinyBertCrossAttnBiGRU(model_name=base_model_name).to(device)
        
        # Load the saved weights
        weights_path = os.path.join(model_path, "full_model.pth")
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback if full_model.pth isn't there but safe tensors are
            # (Though your screenshot shows full_model.pth exists)
            st.warning("full_model.pth not found, trying to load adapter...")
            from peft import PeftModel
            model.backbone = PeftModel.from_pretrained(model.backbone, model_path)
            
        model.eval()
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Main App Interface ---
st.title("🕵️ TinyBERT Fake News Detector")
st.markdown("### Paste a news headline and body to verify its authenticity.")

# Load model
with st.spinner("Loading Model..."):
    model, tokenizer, device = load_model()

# Input columns
col1, col2 = st.columns([1, 1])

with col1:
    headline = st.text_input("📰 Headline", placeholder="Enter the news headline here...")

with col2:
    body = st.text_area("📝 Article Body", placeholder="Paste the article content here...", height=150)

# Predict button
if st.button("🔍 Analyze News"):
    if not headline or not body:
        st.warning("⚠️ Please provide both a Headline and Article Body.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize
            t = tokenizer(headline, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
            b = tokenizer(body, max_length=448, truncation=True, padding='max_length', return_tensors='pt')
            
            # Predict
            with torch.no_grad():
                prob = model(
                    t['input_ids'].to(device),
                    t['attention_mask'].to(device),
                    b['input_ids'].to(device),
                    b['attention_mask'].to(device)
                ).item()
            
            # Display Result
            confidence = prob if prob > 0.5 else 1 - prob
            label = "FAKE" if prob > 0.5 else "REAL"
            
            if label == "FAKE":
                st.markdown(f"""
                    <div class="prediction-box fake">
                        🚨 FAKE NEWS DETECTED<br>
                        <span style="font-size: 18px">Confidence: {confidence:.2%}</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box real">
                        ✅ REAL NEWS VERIFIED<br>
                        <span style="font-size: 18px">Confidence: {confidence:.2%}</span>
                    </div>
                """, unsafe_allow_html=True)

            # Debug Info (Optional)
            with st.expander("See technical details"):
                st.write(f"Raw Model Output (Probability of Fake): {prob:.4f}")
                st.write(f"Device used: {device}")
