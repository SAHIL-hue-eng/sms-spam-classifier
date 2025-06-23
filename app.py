import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam Email Detector with Probability & Emojis")
st.write("Type an email message below and click 'Predict' to check if it's Spam or not.")

message = st.text_area("✉️ Type your email here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        vec_msg = vectorizer.transform([message])
        prediction = model.predict(vec_msg)[0]
        proba = model.predict_proba(vec_msg)[0][prediction] * 100

        if prediction == 1:
            st.error(f"🚨 This is a SPAM message! \n🟡 Confidence: {proba:.2f}%")
            st.markdown("#### ❌📩 🚫")
        else:
            st.success(f"✅ This is a HAM (not spam) message! \n🟢 Confidence: {proba:.2f}%")
            st.markdown("#### 💌📬 ✅")
