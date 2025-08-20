# Backend Agronex

This repository contains the backend for the Agronex project, focused on plant disease detection and management.

## Project Structure
- `app.py`: Main application file (likely a web server or API).
- `model_train.py`: Script for training the plant disease detection model.
- `requirements.txt`: Python dependencies for the project.
- `model/`
  - `class_indices.json`: Mapping of class indices for disease categories.
  - `plant_disease_model.h5`: Trained machine learning model.
- `static/`: Static files (images, CSS, etc.).
- `coverage/`: Coverage reports (if applicable).

## Setup Instructions
1. **Clone the repository**
   ```powershell
   git clone https://github.com/ShubhamG2004/backend_agronex.git
   cd backend_agronex
   ```
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the application**
   ```powershell
   python app.py
   ```

## Usage
- The backend exposes endpoints for plant disease detection and related features.
- You can train the model using `model_train.py` if you wish to update or retrain the model.

## Steps to Train Data and Build the Model

1. **Collect Dataset**
   - Gather images of plant leaves, each labeled with the corresponding disease or healthy class.

2. **Preprocess Data**
   - Resize images to a uniform size.
   - Normalize pixel values.
   - Split the dataset into training and validation sets.

3. **Define Model Architecture**
   - Use deep learning frameworks such as Keras/TensorFlow to define a Convolutional Neural Network (CNN) suitable for image classification.

4. **Train the Model**
   - Feed the training data into the model.
   - Monitor performance on the validation set.
   - Adjust hyperparameters as needed (epochs, batch size, learning rate).

5. **Evaluate and Save Model**
   - Evaluate the trained model's accuracy and loss.
   - Save the trained model as `model/plant_disease_model.h5`.
   - Save the class indices mapping as `model/class_indices.json`.

6. **Retrain (Optional)**
   - To retrain the model, run:

     ```powershell
     python model_train.py
     ```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.
