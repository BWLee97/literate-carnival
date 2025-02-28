import cv2
import numpy as np
import pandas as pd
import streamlit as st
import requests
import os
print(cv2.__version__)
st.set_page_config(layout="wide")
st.title('Floc Image Recognition and Feature Extraction')
with st.container(border=True):
    st.write("This system is only for learning and communication. It is strictly prohibited to use it for commercial purposes. If you have any questions, please contact likx@nankai.edu.cn.")
col_1, col_2 = st.columns(2, border=True)
# 要求上传图片
with col_1:
    st.subheader('Upload Image')
    uploaded_file = st.file_uploader("Please upload a floc image.")
    if uploaded_file is None:
        st.write('**Please upload a floc image.**')
with col_2:
    st.subheader('Select Parameters')
    threshold = st.number_input('Threshold', 0, 300, 100)
    kernel = st.selectbox('Kernel', (3, 5, 7))

if uploaded_file is not None:
    st.success('Successfully upload the image, please tune the parameters for optimal image processing.', icon="✅")
else:
    st.info('Please upload a floc image.', icon="ℹ️")
col_3, col_4 = st.columns(2, border=True)
# 絮体图像
with col_3:
    st.subheader('Image Processing and Recognition')
    tab_31, tab_32 = st.tabs(["Image", "Nomogram"])
    with tab_31:
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()),
                                    dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            gray_image = cv2.cvtColor(opencv_image,
                                      cv2.COLOR_BGR2GRAY)
            ret, thresh_image = cv2.threshold(gray_image,
                                              threshold, 255,
                                              cv2.THRESH_BINARY)
            median_img = cv2.medianBlur(thresh_image, kernel)
            kernel_open = np.ones((kernel, kernel), np.uint8)
            opening = cv2.morphologyEx(median_img,
                                       cv2.MORPH_OPEN,
                                       kernel_open)
            col_31, col_32 = st.columns(2)
            with col_31:
                st.image(opencv_image, channels="BGR",
                         caption="Raw Image",
                         use_container_width=True)
            with col_32:
                st.image(opening, channels="Gray",
                         caption="Processed Image",
                         use_container_width=True)
        else:
            st.warning('Please upload an image.', icon="⚠️")
    with tab_32:
        file_path = "nomogram.jpg"
        if os.path.exists(file_path):
            st.image('nomogram.jpg',
                     caption="Nomogram for identifying effluent quality",
                     use_container_width=True)
        else:
            nomogram_url = "https://gitee.com/LeeBoWen/Floc_Recognition/raw/master/%E5%88%97%E7%BA%BF%E5%9B%BE.png"
            response = requests.get(nomogram_url)
            with open('nomogram.jpg', 'wb') as file:
                file.write(response.content)
            st.image('nomogram.jpg',
                     caption="Nomogram for identifying effluent quality",
                     use_container_width=True)
# 絮体特征
with col_4:
    st.subheader('Floc Image Feature Extraction')
    tab_41, tab_42 = st.tabs(["Local Feature", "Global Feature"])
    with tab_41:
        if uploaded_file is not None:
            contours, hierarchy = cv2.findContours(opening,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            floc_count = len(contours)
            local_data = pd.DataFrame(columns=['Number',
                                               'Area',
                                               'Perimeter',
                                               'Equivalent Diameter'])
            i = 1
            area = []
            length = []
            diameter = []
            height = []
            log_area = []
            log_length = []
            for contour in contours:
                A = cv2.contourArea(contour)  # 面积
                L = cv2.arcLength(contour, closed=True)  # 周长/弧长
                D = 2 * np.sqrt(A / np.pi)  # 等效粒径
                log_area.append(np.log(A))
                log_length.append(np.log(L))
                x, y, w, h = cv2.boundingRect(contour)  # 长、宽
                area.append(A)
                length.append(L)
                diameter.append(D)
                height.append(h)
                new_row = pd.DataFrame([{'Number': i,
                                         'Area': A,
                                         'Perimeter': L,
                                         'Equivalent Diameter': D}])
                local_data = pd.concat([local_data, new_row],
                                       ignore_index=True)
                i += 1
            local_data.index = local_data['Number']
            local_data.drop(columns=['Number'], axis=1, inplace=True)
            st.dataframe(local_data, use_container_width=True)
        else:
            st.warning('Please upload an image.', icon="⚠️")
    with tab_42:
        if uploaded_file is not None:
            global_data = pd.DataFrame(columns=['Floc Number',
                                                'Average Equivalent Diameter',
                                                'Fractal Dimension',
                                                'Image Density'])
            average_area = np.mean(area)  # 平均面积
            average_diameter = np.mean(diameter)  # 平均粒径
            # 分形维数
            z1 = np.polyfit(log_length, log_area, 1)
            p1 = np.poly1d(z1)
            fractal_dimension = p1.coeffs[0]
            floc_pixel_count = 0
            for contour in contours:
                floc_pixel_count += cv2.contourArea(contour)
            total_pixel_count = opening.shape[0] * opening.shape[1]
            image_density = floc_pixel_count / total_pixel_count
            global_data = pd.DataFrame([{'Floc Number': floc_count,
                                         'Average Equivalent Diameter': average_diameter,
                                         'Fractal Dimension': fractal_dimension,
                                         'Image Density': image_density}])
            st.dataframe(global_data,
                         use_container_width=True,
                         hide_index = True)
        else:
            st.warning('Please upload an image.', icon="⚠️")
