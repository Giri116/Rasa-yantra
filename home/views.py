from plotly.offline import plot
from plotly.subplots import make_subplots
import pandas
import plotly.graph_objects as go
from django.shortcuts import render, HttpResponse, redirect
from .forms import ImageUploadForm
from .forms import UploadedImage
from django.core.files.storage import FileSystemStorage
from os import getcwd

# Create your views here.
from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import numpy as np
import base64

def home(request):
    return render(request, 'index.html')

def blog(request):
    return render(request, 'upcoming.html')

def analysis(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            # Retrieve the uploaded image from the form
            uploaded_image = form.cleaned_data['image']
            print(type(uploaded_image))

            # Process the uploaded image using OpenCV
            processed_image, intensities = process_with_opencv(uploaded_image)
            tastes =  ['Madhura','Lavana','Amla','Tikta','Katu','Kashaya']
            intensity = tastes[intensities.index(max(intensities))]
            print(intensity)
             # Encode the processed image as a JPEG in memory
            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_image_bytes = buffer.tobytes()

             # Convert the processed image to a base64-encoded string
            processed_image_base64 = base64.b64encode(processed_image_bytes).decode("utf-8")

            arr = np.array(intensities)
            chart_divs = visualize_with_plotly(arr)
            # Render a template with the processed image
            return render(request, 'results.html', {'processed_image_base64': processed_image_base64,'chart_divs':chart_divs, 'intensity':intensity})

    else:
        form = ImageUploadForm()

    return render(request, 'analysis.html', {'form': form})
# Function to calculate combined RGB intensity percentage for a region of interest (ROI)
def calculate_combined_rgb_percentage(roi):
    # Calculate the total number of pixels in the ROI
    total_pixels = roi.shape[0] * roi.shape[1]

    # Calculate the total intensity as a sum of the intensities in the red, green, and blue channels
    total_intensity = np.sum(roi)

    # Calculate the combined RGB percentage based on the total intensity
    combined_percentage = (total_intensity / (255 * 3 * total_pixels)) * 100

    return combined_percentage


def process_with_opencv(image_file):
    try:
        # Read the uploaded image file as a NumPy array
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Check if the image was loaded successfully
        if img is not None:
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise and improve circle detection
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # Apply the Hough Circle Transform to detect circles in the image
            circles = cv2.HoughCircles(
                gray_blurred,
                cv2.HOUGH_GRADIENT,
                dp=2,  # Inverse ratio of the accumulator resolution to the image resolution
                minDist=120,  # Minimum distance between detected centers
                param1=80,  # Upper threshold for edge detection
                param2=50,  # Threshold for center detection
                minRadius=80,  # Minimum circle radius
                maxRadius=90  # Maximum circle radius (adjust based on your needs)
            )

            # Ensure circles were detected
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                # Sort the circles by their y-coordinate (top to bottom)
                circles = sorted(circles, key=lambda x: x[1])

                # Initialize a list to store intensity values
                intensity_values = []
                # Iterate through detected circles
                for i, (x, y, r) in enumerate(circles):
                    # Define a region of interest (ROI) within a circle
                    roi = img[y - r : y + r, x - r : x + r]

                    # Calculate the combined RGB intensity percentage for the ROI
                    combined_percentage = calculate_combined_rgb_percentage(roi)

                    # Append the intensity value to the list
                    intensity_values.append(combined_percentage)

                    # Label each circle with a numeric value (1, 2, 3, etc.)
                    circle_label = str(i+1)

                    # Draw the circle with the label
                    cv2.circle(img, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(img, circle_label, (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                # Encode the image as JPEG format and return it as a binary response
                _, img_bytes = cv2.imencode(".jpg", img)
                response = HttpResponse(content_type="image/jpeg")
                response.write(img_bytes.tobytes())                    
                # Display the intensity values as separate text output
                for i, intensity in enumerate(intensity_values):
                    print(f"Circle {i + 1}: {intensity:.2f}%")

                # Save the result to a file if needed
                # cv2.imwrite('output.jpg', img)
                

            # Return the processed image as a NumPy array
            return img, intensity_values

    except Exception as e:
        print("Error:", str(e))
        print('Hello')
        return None

def visualize_with_plotly(data):
    # Your data processing and chart generation code here
    # Replace this with your actual data and chart generation code
    print(data.shape)
    ### THIS NEED TO BE CHANGED 
    column_values =  ['Madhura','Lavana','Amla','Tikta','Katu','Kashaya','Sample']
    # df = pandas.DataFrame(columns=column_values,data=data)
    # x = np.arange(1,data.shape[0])

    # chart = go.Pie(values=data, labels=column_values)
    chart = go.Scatterpolar(r=data, theta=column_values, fill='toself')
    pie_chart = go.Pie(values=data, labels=column_values)

    # Create a Plotly figure
    fig = make_subplots(rows=1, cols=2,  specs=[[{'type': 'scatterpolar'}, {'type': 'pie'}]])
    fig.add_trace(chart, row=1, col=1)
    fig.add_trace(pie_chart, row=1, col=2)
    fig.update_layout(title = 'Intensity distribution',width=1000, height=700, paper_bgcolor='rgba(0,90,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # Convert the figure to JSON
    # chart_json = fig.to_json()
    chart_divs = fig.to_html(full_html=False, include_plotlyjs=False)

    return chart_divs