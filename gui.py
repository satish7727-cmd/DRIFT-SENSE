import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import os
import sys
import json
import pandas as pd

# ============================================================
# DRIFT-SENSE
# Professional GUI
# ============================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from detector import rotation_scale_detector


# ============================================================
# COLORS
# ============================================================

BG = "#F3F6FA"
CARD = "#FFFFFF"
NAVY = "#142B4A"
NAVY2 = "#1D3A5F"
BLUE = "#2878D0"
BLUE_DARK = "#1F5FA8"
GREEN = "#20A464"
GREEN_LIGHT = "#E8F7EF"
RED = "#D94A4A"
TEXT = "#17202A"
MUTED = "#6B7785"
BORDER = "#DCE3EA"
WHITE = "#FFFFFF"
LIGHT_BLUE = "#EAF2FB"


# ============================================================
# APPLICATION
# ============================================================

class DriftSenseApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "DRIFT-SENSE | Robust Target Detection"
        )

        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)
        self.root.configure(bg=BG)

        # Images
        self.reference_image = None
        self.test_image = None

        self.reference_path = None
        self.test_path = None

        # Detection
        self.detection_result = None

        self.build_interface()

    # ========================================================
    # MAIN INTERFACE
    # ========================================================

    def build_interface(self):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = tk.Frame(
            self.root,
            bg=NAVY,
            height=90
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(False)

        title_frame = tk.Frame(
            header,
            bg=NAVY
        )

        title_frame.pack(
            side="left",
            padx=30
        )

        tk.Label(
            title_frame,
            text="DRIFT-SENSE",
            font=("DejaVu Sans", 25, "bold"),
            bg=NAVY,
            fg=WHITE
        ).pack(
            anchor="w",
            pady=(12, 0)
        )

        tk.Label(
            title_frame,
            text="Robust Multi-scale & Rotation-aware Target Detection",
            font=("DejaVu Sans", 10),
            bg=NAVY,
            fg="#C8D7E8"
        ).pack(
            anchor="w"
        )

        # System status

        status_frame = tk.Frame(
            header,
            bg=NAVY
        )

        status_frame.pack(
            side="right",
            padx=30
        )

        self.system_status = tk.Label(
            status_frame,
            text="● SYSTEM READY",
            font=("DejaVu Sans", 11, "bold"),
            bg=NAVY,
            fg="#5BE39A"
        )

        self.system_status.pack()

        # ----------------------------------------------------
        # MAIN CONTENT
        # ----------------------------------------------------

        content = tk.Frame(
            self.root,
            bg=BG
        )

        content.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=20
        )

        # ----------------------------------------------------
        # TOP IMAGE AREA
        # ----------------------------------------------------

        image_area = tk.Frame(
            content,
            bg=BG
        )

        image_area.pack(
            fill="both",
            expand=True
        )

        # ====================================================
        # REFERENCE CARD
        # ====================================================

        reference_card = tk.Frame(
            image_area,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        reference_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        tk.Label(
            reference_card,
            text="REFERENCE IMAGE",
            font=("DejaVu Sans", 14, "bold"),
            bg=CARD,
            fg=NAVY
        ).pack(
            pady=(15, 2)
        )

        tk.Label(
            reference_card,
            text="Template / Target",
            font=("DejaVu Sans", 9),
            bg=CARD,
            fg=MUTED
        ).pack()

        self.reference_display = tk.Label(
            reference_card,
            text="No image selected\n\nSelect a reference image",
            font=("DejaVu Sans", 11),
            bg="#EEF2F6",
            fg=MUTED
        )

        self.reference_display.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=15
        )

        self.reference_button = tk.Button(
            reference_card,
            text="＋  Select Reference",
            command=self.select_reference,
            font=("DejaVu Sans", 10, "bold"),
            bg=BLUE,
            fg=WHITE,
            activebackground=BLUE_DARK,
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=8
        )

        self.reference_button.pack(
            pady=(0, 15)
        )

        # ====================================================
        # TEST IMAGE CARD
        # ====================================================

        test_card = tk.Frame(
            image_area,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        test_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=10
        )

        tk.Label(
            test_card,
            text="TEST IMAGE",
            font=("DejaVu Sans", 14, "bold"),
            bg=CARD,
            fg=NAVY
        ).pack(
            pady=(15, 2)
        )

        tk.Label(
            test_card,
            text="Search Image",
            font=("DejaVu Sans", 9),
            bg=CARD,
            fg=MUTED
        ).pack()

        self.test_display = tk.Label(
            test_card,
            text="No image selected\n\nSelect a test image",
            font=("DejaVu Sans", 11),
            bg="#EEF2F6",
            fg=MUTED
        )

        self.test_display.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=15
        )

        self.test_button = tk.Button(
            test_card,
            text="＋  Select Test Image",
            command=self.select_test,
            font=("DejaVu Sans", 10, "bold"),
            bg=BLUE,
            fg=WHITE,
            activebackground=BLUE_DARK,
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=8
        )

        self.test_button.pack(
            pady=(0, 15)
        )

        # ====================================================
        # RESULTS CARD
        # ====================================================

        result_card = tk.Frame(
            image_area,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            width=350
        )

        result_card.pack(
            side="right",
            fill="y",
            padx=(10, 0)
        )

        result_card.pack_propagate(False)

        tk.Label(
            result_card,
            text="DETECTION RESULTS",
            font=("DejaVu Sans", 14, "bold"),
            bg=CARD,
            fg=NAVY
        ).pack(
            pady=(18, 2)
        )

        tk.Label(
            result_card,
            text="Real-time analysis",
            font=("DejaVu Sans", 9),
            bg=CARD,
            fg=MUTED
        ).pack()

        # Score

        score_box = tk.Frame(
            result_card,
            bg=LIGHT_BLUE
        )

        score_box.pack(
            fill="x",
            padx=20,
            pady=18
        )

        tk.Label(
            score_box,
            text="MATCHING SCORE",
            font=("DejaVu Sans", 9, "bold"),
            bg=LIGHT_BLUE,
            fg=MUTED
        ).pack(
            pady=(12, 0)
        )

        self.score_label = tk.Label(
            score_box,
            text="--",
            font=("DejaVu Sans", 30, "bold"),
            bg=LIGHT_BLUE,
            fg=BLUE
        )

        self.score_label.pack(
            pady=(0, 12)
        )

        # Result rows

        self.location_value = self.create_result_row(
            result_card,
            "LOCATION"
        )

        self.scale_value = self.create_result_row(
            result_card,
            "SCALE"
        )

        self.rotation_value = self.create_result_row(
            result_card,
            "ROTATION"
        )

        self.size_value = self.create_result_row(
            result_card,
            "TARGET SIZE"
        )

        # Status

        status_box = tk.Frame(
            result_card,
            bg="#F6F8FA"
        )

        status_box.pack(
            fill="x",
            padx=20,
            pady=15
        )

        self.detection_status = tk.Label(
            status_box,
            text="● WAITING FOR DETECTION",
            font=("DejaVu Sans", 10, "bold"),
            bg="#F6F8FA",
            fg=MUTED
        )

        self.detection_status.pack(
            pady=12
        )

        # ====================================================
        # CONTROL BAR
        # ====================================================

        controls = tk.Frame(
            content,
            bg=BG
        )

        controls.pack(
            fill="x",
            pady=(18, 12)
        )

        # Detect

        self.detect_button = tk.Button(
            controls,
            text="▶  DETECT TARGET",
            command=self.detect,
            font=("DejaVu Sans", 12, "bold"),
            bg=GREEN,
            fg=WHITE,
            activebackground="#188653",
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            padx=30,
            pady=12
        )

        self.detect_button.pack(
            side="left",
            padx=(0, 10)
        )

        # Clear

        tk.Button(
            controls,
            text="✕  CLEAR",
            command=self.clear_all,
            font=("DejaVu Sans", 10, "bold"),
            bg=WHITE,
            fg=TEXT,
            activebackground="#E8EDF2",
            relief="flat",
            cursor="hand2",
            padx=22,
            pady=12
        ).pack(
            side="left",
            padx=5
        )

        # Save

        tk.Button(
            controls,
            text="↓  SAVE RESULT",
            command=self.save_result,
            font=("DejaVu Sans", 10, "bold"),
            bg=WHITE,
            fg=TEXT,
            activebackground="#E8EDF2",
            relief="flat",
            cursor="hand2",
            padx=22,
            pady=12
        ).pack(
            side="left",
            padx=5
        )

        # Robustness

        tk.Button(
            controls,
            text="▣  ROBUSTNESS",
            command=self.show_robustness,
            font=("DejaVu Sans", 10, "bold"),
            bg=WHITE,
            fg=TEXT,
            activebackground="#E8EDF2",
            relief="flat",
            cursor="hand2",
            padx=22,
            pady=12
        ).pack(
            side="right"
        )

        # ====================================================
        # STATUS BAR
        # ====================================================

        footer = tk.Frame(
            self.root,
            bg=NAVY2,
            height=35
        )

        footer.pack(
            fill="x"
        )

        footer.pack_propagate(False)

        self.footer_status = tk.Label(
            footer,
            text="Ready | Select reference and test images",
            font=("DejaVu Sans", 9),
            bg=NAVY2,
            fg="#D4DFEB"
        )

        self.footer_status.pack(
            side="left",
            padx=20
        )

        tk.Label(
            footer,
            text="DRIFT-SENSE  •  Python + OpenCV",
            font=("DejaVu Sans", 9),
            bg=NAVY2,
            fg="#AEBFD1"
        ).pack(
            side="right",
            padx=20
        )

    # ========================================================
    # RESULT ROW
    # ========================================================

    def create_result_row(self, parent, title):

        frame = tk.Frame(
            parent,
            bg=CARD
        )

        frame.pack(
            fill="x",
            padx=20,
            pady=5
        )

        tk.Label(
            frame,
            text=title,
            font=("DejaVu Sans", 9),
            bg=CARD,
            fg=MUTED
        ).pack(
            side="left"
        )

        value = tk.Label(
            frame,
            text="--",
            font=("DejaVu Sans", 10, "bold"),
            bg=CARD,
            fg=TEXT
        )

        value.pack(
            side="right"
        )

        return value

    # ========================================================
    # SELECT REFERENCE
    # ========================================================

    def select_reference(self):

        path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[
                ("Image Files",
                 "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:

            messagebox.showerror(
                "Image Error",
                "Unable to read the selected image."
            )

            return

        self.reference_path = path
        self.reference_image = image

        self.show_image(
            image,
            self.reference_display
        )

        self.footer_status.config(
            text="Reference loaded: " +
            os.path.basename(path)
        )

        self.system_status.config(
            text="● REFERENCE LOADED",
            fg="#5BE39A"
        )

    # ========================================================
    # SELECT TEST
    # ========================================================

    def select_test(self):

        path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[
                ("Image Files",
                 "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:

            messagebox.showerror(
                "Image Error",
                "Unable to read the selected image."
            )

            return

        self.test_path = path
        self.test_image = image

        self.show_image(
            image,
            self.test_display
        )

        self.footer_status.config(
            text="Test image loaded: " +
            os.path.basename(path)
        )

        self.system_status.config(
            text="● IMAGES READY",
            fg="#5BE39A"
        )

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    def show_image(self, image, label):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(rgb)

        # Large preview but keep aspect ratio

        pil.thumbnail(
            (600, 330),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            pil
        )

        label.config(
            image=photo,
            text=""
        )

        label.image = photo

    # ========================================================
    # DETECTION
    # ========================================================

    def detect(self):

        if self.reference_image is None:

            messagebox.showwarning(
                "Reference Image Missing",
                "Please select the reference image first."
            )

            return

        if self.test_image is None:

            messagebox.showwarning(
                "Test Image Missing",
                "Please select the test image first."
            )

            return

        try:

            self.detect_button.config(
                text="⏳  DETECTING...",
                state="disabled",
                bg="#7A8795"
            )

            self.system_status.config(
                text="● PROCESSING...",
                fg="#FFD166"
            )

            self.footer_status.config(
                text="Running multi-scale + rotation detection..."
            )

            self.root.update()

            # Convert to grayscale

            reference_gray = cv2.cvtColor(
                self.reference_image,
                cv2.COLOR_BGR2GRAY
            )

            test_gray = cv2.cvtColor(
                self.test_image,
                cv2.COLOR_BGR2GRAY
            )

            # IMPORTANT:
            # detector expects:
            # search_image = TEST
            # template = REFERENCE

            result = rotation_scale_detector(
                test_gray,
                reference_gray
            )

            self.detection_result = result

            score = result.get("score")
            location = result.get("location")
            angle = result.get("angle")
            scale = result.get("scale")
            size = result.get("size")

            if (
                score is None
                or location is None
                or scale is None
                or angle is None
                or size is None
                or score < 0
            ):

                self.show_failed_detection(
                    result
                )

                return

            # ------------------------------------------------
            # UPDATE RESULT CARDS
            # ------------------------------------------------

            self.score_label.config(
                text=f"{score * 100:.2f}%"
            )

            self.location_value.config(
                text=f"({location[0]}, {location[1]})"
            )

            self.scale_value.config(
                text=f"{scale:.2f}×"
            )

            self.rotation_value.config(
                text=f"{angle:.1f}°"
            )

            self.size_value.config(
                text=f"{size[0]} × {size[1]}"
            )

            self.detection_status.config(
                text="●  TARGET DETECTED",
                fg=GREEN,
                bg=GREEN_LIGHT
            )

            # Draw detection rectangle

            self.show_detection_image(
                location,
                size
            )

            self.system_status.config(
                text="● DETECTION SUCCESS",
                fg="#5BE39A"
            )

            self.footer_status.config(
                text=(
                    f"Detection complete | "
                    f"Score: {score:.6f} | "
                    f"Location: {location}"
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Detection Error",
                str(e)
            )

            self.system_status.config(
                text="● ERROR",
                fg="#FF7777"
            )

            self.footer_status.config(
                text="Detection error"
            )

        finally:

            self.detect_button.config(
                text="▶  DETECT TARGET",
                state="normal",
                bg=GREEN
            )

    # ========================================================
    # SHOW DETECTION IMAGE
    # ========================================================

    def show_detection_image(
        self,
        location,
        size
    ):

        if self.test_image is None:
            return

        x, y = location
        w, h = size

        # Copy image

        image = self.test_image.copy()

        # Draw target rectangle

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 220, 90),
            5
        )

        # Crosshair

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.drawMarker(
            image,
            (center_x, center_y),
            (0, 220, 90),
            cv2.MARKER_CROSS,
            35,
            4
        )

        # Label background

        label_text = "TARGET"

        cv2.putText(
            image,
            label_text,
            (x, max(35, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 220, 90),
            3,
            cv2.LINE_AA
        )

        # Display

        self.show_image(
            image,
            self.test_display
        )

    # ========================================================
    # FAILED DETECTION
    # ========================================================

    def show_failed_detection(
        self,
        result
    ):

        self.score_label.config(
            text="FAILED",
            fg=RED
        )

        self.location_value.config(
            text="--"
        )

        self.scale_value.config(
            text="--"
        )

        self.rotation_value.config(
            text="--"
        )

        self.size_value.config(
            text="--"
        )

        self.detection_status.config(
            text="●  DETECTION FAILED",
            fg=RED,
            bg="#FCEBEC"
        )

        self.system_status.config(
            text="● DETECTION FAILED",
            fg="#FF7777"
        )

        self.footer_status.config(
            text="Target could not be detected"
        )

        messagebox.showwarning(
            "Detection Failed",
            "The target could not be reliably detected."
        )

    # ========================================================
    # CLEAR
    # ========================================================

    def clear_all(self):

        self.reference_image = None
        self.test_image = None

        self.reference_path = None
        self.test_path = None

        self.detection_result = None

        self.reference_display.config(
            image="",
            text="No image selected\n\nSelect a reference image"
        )

        self.test_display.config(
            image="",
            text="No image selected\n\nSelect a test image"
        )

        self.reference_display.image = None
        self.test_display.image = None

        self.score_label.config(
            text="--",
            fg=BLUE
        )

        self.location_value.config(
            text="--"
        )

        self.scale_value.config(
            text="--"
        )

        self.rotation_value.config(
            text="--"
        )

        self.size_value.config(
            text="--"
        )

        self.detection_status.config(
            text="● WAITING FOR DETECTION",
            fg=MUTED,
            bg="#F6F8FA"
        )

        self.system_status.config(
            text="● SYSTEM READY",
            fg="#5BE39A"
        )

        self.footer_status.config(
            text="Ready | Select reference and test images"
        )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    def save_result(self):

        if self.detection_result is None:

            messagebox.showinfo(
                "No Result",
                "Run detection first."
            )

            return

        path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".json",
            filetypes=[
                ("JSON Files", "*.json"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        try:

            result = self.detection_result.copy()

            # Convert tuples to lists for JSON

            if result.get("location") is not None:
                result["location"] = list(
                    result["location"]
                )

            if result.get("size") is not None:
                result["size"] = list(
                    result["size"]
                )

            result["reference_image"] = (
                self.reference_path
            )

            result["test_image"] = (
                self.test_path
            )

            with open(
                path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=4
                )

            messagebox.showinfo(
                "Saved",
                "Detection result saved successfully."
            )

            self.footer_status.config(
                text="Result saved: " +
                os.path.basename(path)
            )

        except Exception as e:

            messagebox.showerror(
                "Save Error",
                str(e)
            )

    # ========================================================
    # ROBUSTNESS
    # ========================================================

    def show_robustness(self):

        csv_path = os.path.join(
            os.path.dirname(__file__),
            "results",
            "robustness_results_corrected.csv"
        )

        if not os.path.exists(csv_path):

            messagebox.showinfo(
                "Robustness Test",
                "No corrected robustness results found.\n\n"
                "Run the robustness test first."
            )

            return

        try:

            df = pd.read_csv(
                csv_path
            )

            window = tk.Toplevel(
                self.root
            )

            window.title(
                "DRIFT-SENSE | Robustness Analysis"
            )

            window.geometry(
                "850x600"
            )

            window.configure(
                bg=BG
            )

            tk.Label(
                window,
                text="ROBUSTNESS ANALYSIS",
                font=("DejaVu Sans", 20, "bold"),
                bg=BG,
                fg=NAVY
            ).pack(
                pady=(20, 5)
            )

            # Summary

            if "Matching Score" in df.columns:
                avg_score = df[
                    "Matching Score"
                ].mean()
            else:
                avg_score = 0

            if "Success" in df.columns:
                success_rate = (
                    df["Success"]
                    .astype(str)
                    .str.lower()
                    .eq("true")
                    .mean()
                    * 100
                )
            else:
                success_rate = 0

            summary = tk.Frame(
                window,
                bg=BG
            )

            summary.pack(
                fill="x",
                padx=25,
                pady=15
            )

            self.robust_card(
                summary,
                "TEST CASES",
                str(len(df))
            )

            self.robust_card(
                summary,
                "AVG SCORE",
                f"{avg_score:.4f}"
            )

            self.robust_card(
                summary,
                "SUCCESS RATE",
                f"{success_rate:.1f}%"
            )

            # Text display

            text_frame = tk.Frame(
                window,
                bg=CARD
            )

            text_frame.pack(
                fill="both",
                expand=True,
                padx=25,
                pady=10
            )

            text = tk.Text(
                text_frame,
                font=("DejaVu Sans Mono", 10),
                bg=CARD,
                fg=TEXT,
                relief="flat"
            )

            text.pack(
                fill="both",
                expand=True,
                padx=15,
                pady=15
            )

            text.insert(
                "1.0",
                df.to_string(index=False)
            )

            text.config(
                state="disabled"
            )

        except Exception as e:

            messagebox.showerror(
                "Robustness Error",
                str(e)
            )

    # ========================================================
    # ROBUSTNESS CARD
    # ========================================================

    def robust_card(
        self,
        parent,
        title,
        value
    ):

        card = tk.Frame(
            parent,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        card.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5
        )

        tk.Label(
            card,
            text=title,
            font=("DejaVu Sans", 9, "bold"),
            bg=CARD,
            fg=MUTED
        ).pack(
            pady=(12, 2)
        )

        tk.Label(
            card,
            text=value,
            font=("DejaVu Sans", 20, "bold"),
            bg=CARD,
            fg=BLUE
        ).pack(
            pady=(0, 12)
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DriftSenseApp(
        root
    )

    root.mainloop()