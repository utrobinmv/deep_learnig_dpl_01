import streamlit as st
import cv2
import numpy as np

from test_my import recognition_image

def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

st.set_page_config(page_title="Демо: Поиск карты на картинке!") #, page_icon=None, layout='centered', initial_sidebar_state='auto')

st.sidebar.title("Поиск карты")

list_value = ['Henderson', 'Сенат', 'Карусель', 'Загрузить свое фото...']

photos_files = ['20201031_172457.jpg','IMG_4233.jpg','IMG_5219.jpg']

photo_path = 'data/test_cards/images/'

select_value = st.sidebar.selectbox("Распознавание изображения: ", list_value,key = 'sidebar_01')

st.markdown('Выбор сделан: ' + select_value)

select_index = list_value.index(select_value)

recognition_on = False

if select_index > 2:
    uploaded_file = st.file_uploader("Выберите файл карты:", accept_multiple_files = False)
    
    recognition_on = False

    if uploaded_file is not None:
        recognition_on = True
        if st.button('Очистить загруженное фото', key="button_01"):
            uploaded_file = None
            recognition_on = False
            
            #rerun()
    else:
        st.markdown('Загрузить фото:')
        st.markdown('Выберите файл фотографии с дисконтной картой')
        st.markdown('Важно: Для лучшего качества! Загружаемая картинка должна быть разрешением 840х840')
        st.markdown('и пропорции изображения карты не должны быть нарушены.')
    
    if uploaded_file is not None:
        recognition_on = True
        st.markdown('Файл загружен: ')
        st.markdown(str(uploaded_file))
        
        jpeg_image = uploaded_file.getvalue()
        img_array = np.asarray(bytearray(jpeg_image), dtype=np.uint8)
        np_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
   
        
        
else:
    recognition_on = True
    #st.markdown('Показать фото!')
    
    np_image = cv2.imread(photo_path + photos_files[select_index])
    
    #st.image(np_image, caption = "Фотография исходная", use_column_width = True, channels='BGR')
    
    
if recognition_on == True:
    st.image(np_image, caption = "Фотография исходная", use_column_width = True, channels='BGR')
    
    if st.sidebar.button('Распознать фото', key="button_02"):
        st.sidebar.markdown('Кнопка нажата!')
        st.sidebar.markdown('Ожидайте, идет распознавание...')
        st.sidebar.markdown('Фото появится ниже...')
        new_image = recognition_image(np_image)
        
        st.image(new_image, caption = "Фотография распознана", use_column_width = True, channels='BGR')
        st.sidebar.markdown('Фото появилось! Распознавание окончено!')