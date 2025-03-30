import cv2
import tkinter as tk
from tkinter import messagebox
import os
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import time

class CameraApp:
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Raspberry Pi AI Camera Capture")
        self.root.geometry("800x600")
        
        # Create capture button
        self.capture_button = tk.Button(
            self.root,
            text="Capture Image",
            command=self.capture_image,
            font=("Arial", 14),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        self.capture_button.pack(pady=20)
        
        # Create status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to capture",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=10)
        
        # Create directory for saving images if it doesn't exist
        self.save_dir = "captured_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Initialize camera
        try:
            self.picam2 = Picamera2()
            
            # Configure camera
            camera_config = self.picam2.create_preview_configuration(
                main={"size": (640, 480)},
                lores={"size": (320, 240)},
                display="lores"
            )
            self.picam2.configure(camera_config)
            self.picam2.start()
            
            # Start video preview
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not initialize camera: {str(e)}")
            self.root.quit()
            return
    
    def capture_image(self):
        """Capture and save image from camera"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"image_{timestamp}.jpg")
            
            # Capture image
            self.picam2.capture_file(filename, use_video_port=True)
            
            # Update status
            self.status_label.config(text=f"Image saved: {filename}")
            messagebox.showinfo("Success", "Image captured and saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture image: {str(e)}")
    
    def update_preview(self):
        """Update the video preview"""
        try:
            # Capture frame for preview
            frame = self.picam2.capture_array("lores")
            
            # Convert frame to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tk = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
            
            # Update preview label
            if not hasattr(self, 'preview_label'):
                self.preview_label = tk.Label(self.root, image=frame_tk)
                self.preview_label.pack(pady=10)
            else:
                self.preview_label.config(image=frame_tk)
            
            # Keep reference to prevent garbage collection
            self.preview_label.image = frame_tk
            
        except Exception as e:
            print(f"Preview error: {str(e)}")
        
        # Schedule next update
        self.root.after(10, self.update_preview)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup when application closes"""
        if hasattr(self, 'picam2'):
            self.picam2.stop()

if __name__ == "__main__":
    app = CameraApp()
    app.run() 