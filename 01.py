import streamlit as st
import pickle
import pandas as pd
import joblib

# 加载模型
model = joblib.load('women_depression_rf_model.pkl')


# 预测函数
def make_prediction(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Streamlit 应用界面
st.title('抑郁症预测模型')

# 创建输入控件
age = st.number_input('Age')
major_life_events = st.number_input('Major life events')
negative_thoughts_behaviors = st.number_input('Negative thoughts/behaviors')
number_suicide_attempts = st.number_input('Number of suicide attempts')
ipaq_level = st.number_input('IPAQ level')
bmi = st.number_input('BMI')
sleep_quality = st.number_input('Sleep quality')
perceived_stress = st.number_input('Perceived stress')
hopelessness = st.number_input('Hopelessness')
loneliness = st.number_input('Loneliness')
resilience = st.number_input('Resilience')
alexithymia = st.number_input('Alexithymia')
problem_focused_coping = st.number_input('Problem-focused coping')
emotion_focused_coping = st.number_input('Emotion-focused coping')
self_esteem = st.number_input('Self-esteem')
rumination = st.number_input('Rumination')
emotion_regulation = st.number_input('Emotion regulation')
borderline_personality = st.number_input('Borderline personality')
care = st.number_input('Care')
overprotection = st.number_input('Overprotection')

# 加载模型
# model = load_model('women_depression_rf_model.pkl')

# 当用户点击预测按钮时，显示预测结果
if st.button('预测'):
    # 创建输入数据的 DataFrame
    input_data = pd.DataFrame([[age, major_life_events, negative_thoughts_behaviors, number_suicide_attempts,
                               ipaq_level, bmi, sleep_quality, perceived_stress, hopelessness, loneliness,
                               resilience, alexithymia, problem_focused_coping, emotion_focused_coping,
                               self_esteem, rumination, emotion_regulation, borderline_personality,
                               care, overprotection]],
                              columns=['Age', 'Major life events', 'Negative thoughts/behaviors', 'Number of suicide attempts',
                                       'IPAQ level', 'BMI', 'Sleep quality', 'Perceived stress', 'Hopelessness',
                                       'Loneliness', 'Resilience', 'Alexithymia', 'Problem-focused coping',
                                       'Emotion-focused coping', 'Self-esteem', 'Rumination', 'Emotion regulation',
                                       'Borderline personality', 'Care', 'Overprotection'])

    # 进行预测
    prediction = make_prediction(model, input_data)
    st.write('抑郁预测结果:', prediction)
