import cv2
import tkinter as tk
from tkinter import messagebox
import os
from datetime import datetime
import time

class CameraApp:
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Raspberry Pi Camera Capture")
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
        self.cap = cv2.VideoCapture(0)  # Use 0 for default camera
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            self.root.quit()
            return
        
        # Start video preview
        self.update_preview()
        
    def capture_image(self):
        """Capture and save image from camera"""
        ret, frame = self.cap.read()
        if ret:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"image_{timestamp}.jpg")
            
            # Save image
            cv2.imwrite(filename, frame)
            
            # Update status
            self.status_label.config(text=f"Image saved: {filename}")
            messagebox.showinfo("Success", "Image captured and saved successfully!")
        else:
            messagebox.showerror("Error", "Failed to capture image!")
    
    def update_preview(self):
        """Update the video preview"""
        ret, frame = self.cap.read()
        if ret:
            # Resize frame to fit window
            frame = cv2.resize(frame, (640, 480))
            
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
        
        # Schedule next update
        self.root.after(10, self.update_preview)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup when application closes"""
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    app = CameraApp()
    app.run() 