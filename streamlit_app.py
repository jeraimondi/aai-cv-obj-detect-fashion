import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
from ultralytics import YOLO

# set session state variables for persistence
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None

if 'confidence' not in st.session_state:
    st.session_state.confidence = 60 / 100

if 'iou_thresh' not in st.session_state:
    st.session_state.iou_thresh = 20 / 100

# page layout
st.set_page_config(
    page_title='Fashion Object Detection',  # set page title
    page_icon='👜',  # set page icon
    layout='wide',  # set layout to wide
    initial_sidebar_state='expanded',  # expand sidebar by default
)


# sidebar content
with st.sidebar:

    @st.cache_data
    def load_image(image_file):
        image = Image.open(image_file)
        image = image.convert('RGB')
        return image


    st.header('Image Selection')  # add header to sidebar

    # file uploader
    with st.form('file-upload-form', clear_on_submit=True):
        st.session_state.uploaded_img = st.file_uploader('Upload an image...', type='jpg', key=1)
        submitted = st.form_submit_button('Submit/Clear Upload')

    # load image using function with PIL
    if type(st.session_state.uploaded_img) is st.runtime.uploaded_file_manager.UploadedFile:
        st.session_state.uploaded_img = load_image(st.session_state.uploaded_img)

    # slider to select model confidence
    st.session_state.confidence = float(st.slider(
        'Select Model Confidence', 20, 100, 60)) / 100

    # slider to select IOU threshold
    st.session_state.iou_thresh = float(st.slider(
        'Select IOU Threshold', 0, 100, 20)) / 100

# create main page heading
st.title('Fashion Object Detection')
st.caption('Select a sample fashion photo, or upload your own! - Recommended :blue[600 Height x 400 Width].')
st.caption('Afterwards, click the :blue[Detect Objects] button to see the results.')

# create three columns on the main page
col1, col2, col3 = st.columns(3)

# col 1 containing image samples
with col1:
    sample_img = image_select(
        label='Select a sample fashion photo',
        images=[
            'data/sample_images/1.jpg',
            'data/sample_images/2.jpg',
            'data/sample_images/3.jpg',
            'data/sample_images/4.jpg',
            'data/sample_images/5.jpg',
            'data/sample_images/6.jpg',
        ],
        captions=['Sample #1', 'Sample #2', 'Sample #3', 'Sample #4', 'Sample #5', 'Sample #6'],
        use_container_width=False
    )

# col 2 containing input/source image
with col2:
    @st.cache_data
    def display_img_col2(image):
        with st.container(border=True):
            if source_img is not None:
                # add the uploaded image to col2 with a caption
                st.image(image,
                         caption='Input',
                         use_column_width=True
                         )

    # obtain image from file uploader, otherwise load sample image
    if st.session_state.uploaded_img is not None:
        source_img = st.session_state.uploaded_img
        st.session_state.uploaded_img = None
    elif sample_img is not None:
        source_img = sample_img
        sample_img = None

    # call function to display the image in col 2
    display_img_col2(source_img)

# col 3 contains the result image as well as modeling logic
with col3:
    @st.cache_resource
    def load_model(path):
        return YOLO(path)

    @st.cache_data
    def run_model(inputs):
        return model(inputs)

    # path to pretrained model
    model_path = 'models/best.pt'

    try:
        # load model
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Check the specified path: {model_path}.")
        st.error(e)

    try:
        # sidebar detect objects button
        if st.sidebar.button('Detect Objects'):

            # clear session state to avoid image being used more than once
            st.session_state.uploaded_img = None

            # perform object detection using model
            results = model.predict(source_img,
                                    save=False,
                                    imgsz=(608, 416),  # image size must be multiple of max stride 32
                                    conf=st.session_state.confidence,  # confidence selected by slider
                                    iou=st.session_state.iou_thresh  # IOU threshold selected by slider
                                    )

            # plot model results, and convert to RGB PIL image
            for r in results:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])

            # display result image in col 3
            with st.container(border=True):
                st.image(im,
                         caption='Result',
                         use_column_width=True
                         )
                source_img = None

    except Exception as e:
        st.error(f"Error encountered during model inference.")
        st.error(e)
