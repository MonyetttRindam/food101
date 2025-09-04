import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import time

# --- Page Config ---
st.set_page_config(
    page_title="Food-101 Classifier",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Enhanced Styling ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FFF8E7 0%, #F5F1E8 100%);
        color: #2D3748;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #FF8A80 0%, #FFB74D 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0;
        font-weight: 400;
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 2px dashed #E2E8F0;
        text-align: center;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .top-prediction {
        background: linear-gradient(135deg, #4CAF50, #81C784);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .alt-prediction {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #FF8A80;
        font-weight: 500;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF8A80, #FFB74D);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        box-shadow: 0 4px 15px rgba(255, 138, 128, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 138, 128, 0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    
    .dataset-info {
        background: linear-gradient(135deg, #FFECB3, #FFCCBC);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #FF8A80;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        min-width: 120px;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #FF8A80;
        display: block;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #FF8A80;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🍽️ Food-101 Classifier</h1>
    <p>AI-Powered Food Recognition • Identifikasi 101 Jenis Makanan dengan Teknologi Deep Learning CNN by MonyetttRindam</p>
</div>
""", unsafe_allow_html=True)

# --- Stats Section ---
st.markdown("""
<div class="stats-container">
    <div class="stat-item">
        <span class="stat-number">101</span>
        <div class="stat-label">Jenis Makanan</div>
    </div>
    <div class="stat-item">
        <span class="stat-number">72%</span>
        <div class="stat-label">Akurasi Model</div>
    </div>
    <div class="stat-item">
        <span class="stat-number">100k</span>
        <div class="stat-label">Gambar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Informasi Dataset Food-101 ---
with st.expander("📊 Tentang Dataset Food-101", expanded=True):
    st.markdown("""
    <div class="dataset-info">
    <h4>🍕 Food-101 Dataset</h4>
    <p>Dataset Food-101 adalah kumpulan data besar yang berisi gambar 101 kategori makanan yang berbeda. 
    Setiap kategori memiliki 1000 gambar, dengan total 101.000 gambar. Untuk setiap kelas, 
    250 gambar yang telah ditinjau secara manual disediakan sebagai data uji.</p>
    
    <h5>📁 Karakteristik Dataset:</h5>
    <ul>
        <li>101 kategori makanan dari berbagai negara dan budaya</li>
        <li>1000 gambar per kategori (750 untuk pelatihan, 250 untuk pengujian)</li>
        <li>Gambar dengan resolusi bervariasi, tetapi semua diubah ukurannya menjadi 128x128 piksel untuk pelatihan model</li>
        <li>Mencakup berbagai hidangan populer seperti pizza, sushi, hamburger, steak, dll.</li>
    </ul>
    
    <h5>🎯 Tujuan Dataset:</h5>
    <p>Dataset ini dibuat untuk memajukan penelitian dalam klasifikasi gambar makanan dan 
    pengenalan pola visual dalam domain kuliner. Model yang dilatih pada dataset ini 
    dapat digunakan dalam aplikasi pelacakan makanan, rekomendasi resep, dan analisis kebiasaan makan.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Cara Penggunaan (seperti pada aplikasi Cats vs Dogs) ---
with st.expander("ℹ️ Cara Penggunaan", expanded=False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff9e6 0%, #fff4f4 100%); 
                padding: 1.8rem; border-radius: 15px; border-left: 5px solid #ff9a3c;
                margin-bottom: 1.8rem;">
    <h4 style="color: #5d4037;">📋 Cara Penggunaan:</h4>
    <ul>
        <li>Klik tombol "Browse files" untuk mengunggah gambar makanan</li>
        <li>Pastikan gambar menunjukkan makanan dengan jelas</li>
        <li>Tunggu sebentar hingga sistem menganalisis gambar</li>
        <li>Lihat hasil prediksi dan tingkat kepercayaan sistem</li>
    </ul>
    
    <h4 style="color: #5d4037;">💡 Tips untuk hasil terbaik:</h4>
    <ul>
        <li>Gunakan gambar dengan makanan menghadap ke depan dan pencahayaan yang baik</li>
        <li>Hindarkan gambar yang blur atau gelap</li>
        <li>Gambar dengan latar belakang sederhana lebih mudah dikenali</li>
        <li>Pastikan makanan adalah salah satu dari 101 kategori yang didukung</li>
        <li>Format yang didukung: JPG, JPEG, PNG</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Load model with caching ---
@st.cache_resource
def load_food_model():
    try:
        model_path = hf_hub_download(
            repo_id="MonyetttRindam/foof101abil",
            filename="Food 101.h5"
        )
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# --- Load model ---
with st.spinner("🔄 Loading AI model..."):
    model = load_food_model()

if model is None:
    st.error("❌ Gagal memuat model. Silakan refresh halaman.")
    st.stop()

# --- Mapping index ke label dengan nama yang lebih friendly ---
labels_dict = {
    'apple_pie': 0, 'baby_back_ribs': 1, 'baklava': 2, 'beef_carpaccio': 3, 'beef_tartare': 4,
    'beet_salad': 5, 'beignets': 6, 'bibimbap': 7, 'bread_pudding': 8, 'breakfast_burrito': 9,
    'bruschetta': 10, 'caesar_salad': 11, 'cannoli': 12, 'caprese_salad': 13, 'carrot_cake': 14,
    'ceviche': 15, 'cheese_plate': 16, 'cheesecake': 17, 'chicken_curry': 18, 'chicken_quesadilla': 19,
    'chicken_wings': 20, 'chocolate_cake': 21, 'chocolate_mousse': 22, 'churros': 23, 'clam_chowder': 24,
    'club_sandwich': 25, 'crab_cakes': 26, 'creme_brulee': 27, 'croque_madame': 28, 'cup_cakes': 29,
    'deviled_eggs': 30, 'donuts': 31, 'dumplings': 32, 'edamame': 33, 'eggs_benedict': 34,
    'escargots': 35, 'falafel': 36, 'filet_mignon': 37, 'fish_and_chips': 38, 'foie_gras': 39,
    'french_fries': 40, 'french_onion_soup': 41, 'french_toast': 42, 'fried_calamari': 43, 'fried_rice': 44,
    'frozen_yogurt': 45, 'garlic_bread': 46, 'gnocchi': 47, 'greek_salad': 48, 'grilled_cheese_sandwich': 49,
    'grilled_salmon': 50, 'guacamole': 51, 'gyoza': 52, 'hamburger': 53, 'hot_and_sour_soup': 54,
    'hot_dog': 55, 'huevos_rancheros': 56, 'hummus': 57, 'ice_cream': 58, 'lasagna': 59, 'lobster_bisque': 60,
    'lobster_roll_sandwich': 61, 'macaroni_and_cheese': 62, 'macarons': 63, 'miso_soup': 64, 'mussels': 65,
    'nachos': 66, 'omelette': 67, 'onion_rings': 68, 'oysters': 69, 'pad_thai': 70, 'paella': 71,
    'pancakes': 72, 'panna_cotta': 73, 'peking_duck': 74, 'pho': 75, 'pizza': 76, 'pork_chop': 77,
    'poutine': 78, 'prime_rib': 79, 'pulled_pork_sandwich': 80, 'ramen': 81, 'ravioli': 82,
    'red_velvet_cake': 83, 'risotto': 84, 'samosa': 85, 'sashimi': 86, 'scallops': 87, 'seaweed_salad': 88,
    'shrimp_and_grits': 89, 'spaghetti_bolognese': 90, 'spaghetti_carbonara': 91, 'spring_rolls': 92,
    'steak': 93, 'strawberry_shortcake': 94, 'sushi': 95, 'tacos': 96, 'takoyaki': 97, 'tiramisu': 98,
    'tuna_tartare': 99, 'waffles': 100
}

idx_to_label = {v: k for k, v in labels_dict.items()}

# Function to format food names
def format_food_name(name):
    return name.replace('_', ' ').title()

# Function to get confidence level description
def get_confidence_description(prob):
    if prob >= 0.9:
        return "Sangat Yakin", "🎯"
    elif prob >= 0.7:
        return "Yakin", "✅"
    elif prob >= 0.5:
        return "Cukup Yakin", "🤔"
    else:
        return "Kurang Yakin", "❓"

# --- Upload Section ---
st.markdown("""
<div class="upload-section">
    <h3 style="color: #FF8A80; margin-bottom: 1rem;">📸 Upload Gambar Makanan</h3>
    <p style="color: #666; margin-bottom: 1rem;">Drag & drop gambar atau klik tombol di bawah untuk memilih file</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload gambar makanan dalam format JPG, JPEG, atau PNG (maksimal 200MB)"
)

# --- Main Content ---
if uploaded_file is not None:
    try:
        # Load and display image
        img = Image.open(uploaded_file)
        
        # Layout
        col1, col2 = st.columns([1.3, 1], gap="large")
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(
                img, 
                caption=f'📷 {uploaded_file.name}', 
                use_container_width=True
            )
            
            # Image info
            width, height = img.size
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px; font-size: 0.9rem;">
            📋 <strong>Info Gambar:</strong> {width}x{height}px • {file_size:.1f}KB
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Loading animation
            with st.spinner(""):
                st.markdown("""
                <div class="loading-spinner">
                    <div class="spinner"></div>
                </div>
                <p style="text-align: center; color: #666; margin-top: 1rem;">
                    🤖 AI sedang menganalisis gambar...
                </p>
                """, unsafe_allow_html=True)
                
                # Add small delay for better UX
                time.sleep(1.5)
                
                # Preprocess image
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Predict
                preds = model.predict(img_array, verbose=0)
                
            # Clear loading animation
            st.empty()
            
            # Get top 3 predictions
            top3_idx = preds[0].argsort()[-3:][::-1]
            top3_labels = [idx_to_label[i] for i in top3_idx]
            top3_probs = [preds[0][i] for i in top3_idx]
            
            # Prediction results
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Hasil Prediksi")
            
            # Top prediction
            top_food = format_food_name(top3_labels[0])
            top_prob = top3_probs[0]
            confidence_desc, confidence_icon = get_confidence_description(top_prob)
            
            st.markdown(f"""
            <div class="top-prediction">
                <div style="font-size: 1.3rem; margin-bottom: 0.5rem;">
                    🏆 <strong>{top_food}</strong>
                </div>
                <div style="font-size: 1.1rem;">
                    {confidence_icon} {top_prob*100:.1f}% • {confidence_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Alternative predictions
            st.markdown("**🔍 Prediksi Alternatif:**")
            for i, (label, prob) in enumerate(zip(top3_labels[1:], top3_probs[1:]), 2):
                food_name = format_food_name(label)
                st.markdown(f"""
                <div class="alt-prediction">
                    <strong>#{i} {food_name}</strong> • {prob*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional info
            if top_prob < 0.5:
                st.warning("⚠️ Tingkat kepercayaan rendah. Coba gunakan gambar yang lebih jelas atau pastikan makanan terlihat dengan baik.")
            elif top_prob > 0.9:
                st.success("🎉 Prediksi dengan tingkat kepercayaan sangat tinggi!")
            
            # Action buttons
            st.markdown("---")
            if st.button("🔄 Analisis Gambar Lain", key="new_analysis"):
                st.rerun()
                
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}")
        st.info("💡 Pastikan file yang diupload adalah gambar yang valid.")

else:
    # Welcome message when no file uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #666;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🍽️</div>
        <h3>Siap Mengidentifikasi Makanan Anda!</h3>
        <p>Upload gambar makanan untuk mendapatkan prediksi AI yang akurat</p>
        <div style="margin-top: 2rem; font-size: 0.9rem; color: #888;">
            Didukung oleh teknologi Deep Learning • Dataset Food-101
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #888; font-size: 0.9rem;">
    🚀 Powered by TensorFlow & Streamlit • Food-101 Dataset<br>
    Made by MonyetttRindam
</div>
""", unsafe_allow_html=True)