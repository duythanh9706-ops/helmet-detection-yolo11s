"""Ứng dụng giao diện desktop cho demo Helmet / No Helmet với YOLO11s.

Khởi động: python gui.py   hoặc nhấp đúp run_gui.bat trên Windows.
Tkinter/Pillow đảm nhiệm giao diện; OpenCV và YOLO xử lý trong thread nền.
"""
from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread
import os
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from src.common import DEFAULT_CONF, DEFAULT_MODEL
from src.gui_engine import GuiConfig, process

BASE = Path(__file__).resolve().parent

BG = "#0B1020"
SIDEBAR = "#111B30"
PANEL = "#151F33"
PANEL_ALT = "#1A2840"
BORDER = "#2C3E58"
TEXT = "#F1F5FC"
MUTED = "#9CB0CB"
BLUE = "#4F8DFF"
BLUE_DARK = "#2466DF"
GREEN = "#3DE3B2"
RED = "#FF748A"
YELLOW = "#F9C56D"
FONT = "Segoe UI"


def resolve_path(raw: str) -> Path:
    path = Path(raw.strip().strip('"')).expanduser()
    return path if path.is_absolute() else BASE / path


def ui_label(parent, text, size=10, color=TEXT, bold=False, bg=None, **kwargs):
    return tk.Label(parent, text=text, font=(FONT, size, "bold" if bold else "normal"),
                    fg=color, bg=bg or parent.cget("bg"), **kwargs)


def ui_button(parent, text, command, *, bg=BLUE, fg="white", padx=12, pady=10, **kwargs):
    return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                     activebackground=BLUE_DARK if bg == BLUE else PANEL_ALT,
                     activeforeground=fg, relief="flat", bd=0, cursor="hand2",
                     font=(FONT, 10, "bold"), padx=padx, pady=pady,
                     highlightthickness=0, **kwargs)


class HelmetApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Helmet AI | Nhận diện mũ bảo hiểm – YOLO11s")
        self.geometry("1240x780")
        self.minsize(990, 680)
        self.configure(bg=BG)
        self.mode = "image"
        self.source_by_mode = {"image": "test_images", "video": "test_videos/demo.mp4"}
        self.is_running = False
        self.worker = None
        self.stop_event = Event()
        self.events = queue.Queue()
        self.frame_lock = Lock()
        self.latest_frame = None
        self.frame_data = None
        self.photo = None
        self.closing = False

        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.source_var = tk.StringVar(value="test_images")
        self.camera_var = tk.StringVar(value="0")
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.record_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.input_title_var = tk.StringVar(value="Ảnh đầu vào")
        self.model_info_var = tk.StringVar()
        self.frame_info_var = tk.StringVar(value="Chưa có dữ liệu đầu vào")
        self.count_helmet_var = tk.StringVar(value="0")
        self.count_nohelmet_var = tk.StringVar(value="0")
        self.count_fps_var = tk.StringVar(value="—")
        self.count_processed_var = tk.StringVar(value="0")
        self.conf_label_var = tk.StringVar(value=f"{DEFAULT_CONF:.2f}")

        self._build()
        self._select_mode("image")
        self._update_model_status()
        self.model_var.trace_add("write", lambda *_: self._update_model_status())
        self.conf_var.trace_add("write", lambda *_: self.conf_label_var.set(f"{self.conf_var.get():.2f}"))
        self.after(50, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self):
        # Header
        top = tk.Frame(self, bg=SIDEBAR, height=81)
        top.pack(fill="x")
        top.pack_propagate(False)
        title_area = tk.Frame(top, bg=SIDEBAR)
        title_area.pack(side="left", padx=26, pady=14)
        ui_label(title_area, "HELMET AI", 20, TEXT, True, SIDEBAR).pack(anchor="w")
        ui_label(title_area, "HỆ THỐNG NHẬN DIỆN MŨ BẢO HIỂM · YOLO11s", 9, MUTED, False, SIDEBAR).pack(anchor="w")
        self.header_status = ui_label(top, "●  SẴN SÀNG", 10, GREEN, True, SIDEBAR)
        self.header_status.pack(side="right", padx=24)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Left sidebar
        sidebar = tk.Frame(body, bg=SIDEBAR, width=207)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ui_label(sidebar, "CHẾ ĐỘ NHẬN DIỆN", 9, MUTED, True, SIDEBAR).pack(anchor="w", padx=17, pady=(27, 13))
        self.mode_buttons = {}
        for key, label in [("image", "▧   Ảnh tĩnh"), ("video", "▶   Video"), ("camera", "◉   Camera trực tiếp")]:
            button = ui_button(sidebar, label, lambda name=key: self._select_mode(name),
                               bg=SIDEBAR, fg=MUTED, padx=18, pady=15, anchor="w")
            button.pack(fill="x", padx=9, pady=4)
            self.mode_buttons[key] = button
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=17, pady=(27, 17))
        ui_label(sidebar, "HƯỚNG DẪN", 9, MUTED, True, SIDEBAR).pack(anchor="w", padx=17)
        for line in ("1  Chọn chế độ", "2  Chọn file model .pt", "3  Chọn ảnh/video", "4  Nhấn Bắt đầu"):
            ui_label(sidebar, line, 10, MUTED, bg=SIDEBAR).pack(anchor="w", padx=17, pady=7)
        ui_label(sidebar, "Không cần dataset train", 9, GREEN, bg=SIDEBAR,
                 wraplength=175, justify="left").pack(side="bottom", pady=(0, 26))

        # Right settings panel
        settings = tk.Frame(body, bg=SIDEBAR, width=320)
        settings.pack(side="right", fill="y")
        settings.pack_propagate(False)
        self.settings_panel = settings
        ui_label(settings, "THIẾT LẬP DEMO", 14, TEXT, True, SIDEBAR).pack(anchor="w", padx=19, pady=(24, 17))

        ui_label(settings, "MODEL YOLO (.PT)", 9, MUTED, True, SIDEBAR).pack(anchor="w", padx=19)
        modelrow = tk.Frame(settings, bg=SIDEBAR)
        modelrow.pack(fill="x", padx=19, pady=(8, 2))
        self.model_entry = self._entry(modelrow, self.model_var)
        self.model_entry.pack(side="left", fill="x", expand=True, ipady=8)
        ui_button(modelrow, "...", self._browse_model, bg=PANEL_ALT, padx=12, pady=8).pack(side="left", padx=(5, 0))
        ui_label(settings, "", 9, MUTED, bg=SIDEBAR, textvariable=self.model_info_var,
                 wraplength=277, justify="left").pack(anchor="w", padx=19, pady=(5, 18))

        ui_label(settings, "NGUỒN DỮ LIỆU", 9, MUTED, True, SIDEBAR).pack(anchor="w", padx=19)
        self.source_block = tk.Frame(settings, bg=SIDEBAR)
        self.source_block.pack(fill="x", padx=19, pady=(7, 10))
        ui_label(self.source_block, "Ảnh đầu vào", 10, TEXT, bg=SIDEBAR,
                 textvariable=self.input_title_var).pack(anchor="w", pady=(0, 7))
        self.source_entry = self._entry(self.source_block, self.source_var)
        self.source_entry.pack(fill="x", ipady=9)
        file_buttons = tk.Frame(self.source_block, bg=SIDEBAR)
        file_buttons.pack(fill="x", pady=(7, 0))
        self.browse_source_button = ui_button(file_buttons, "Chọn file", self._browse_source, bg=PANEL_ALT,
                                              padx=12, pady=9)
        self.browse_source_button.pack(side="left", fill="x", expand=True)
        self.browse_folder_button = ui_button(file_buttons, "Chọn thư mục", self._browse_folder,
                                              bg=PANEL_ALT, padx=9, pady=9)
        self.browse_folder_button.pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.camera_block = tk.Frame(settings, bg=SIDEBAR)
        ui_label(self.camera_block, "ID CAMERA", 10, TEXT, bg=SIDEBAR).pack(anchor="w", pady=(0, 6))
        self.camera_entry = self._entry(self.camera_block, self.camera_var)
        self.camera_entry.pack(fill="x", ipady=9)
        self.record_checkbox = tk.Checkbutton(self.camera_block, text="Ghi video camera (.mp4)",
                                              variable=self.record_var, bg=SIDEBAR, fg=TEXT,
                                              selectcolor=PANEL, activebackground=SIDEBAR,
                                              activeforeground=TEXT, font=(FONT, 10),
                                              anchor="w")
        self.record_checkbox.pack(anchor="w", pady=(12, 10))

        self.conf_heading = ui_label(settings, "NGƯỠNG TIN CẬY (CONFIDENCE)", 9, MUTED, True, SIDEBAR)
        self.conf_heading.pack(anchor="w", padx=19, pady=(18, 0))
        slider_row = tk.Frame(settings, bg=SIDEBAR)
        slider_row.pack(fill="x", padx=14, pady=(6, 16))
        self.conf_scale = tk.Scale(slider_row, from_=0.1, to=0.9, resolution=0.01,
                                   orient="horizontal", variable=self.conf_var,
                                   bg=SIDEBAR, fg=TEXT, troughcolor=BORDER,
                                   activebackground=BLUE, highlightthickness=0,
                                   bd=0, showvalue=False, length=215,
                                   sliderrelief="flat")
        self.conf_scale.pack(side="left", fill="x", expand=True)
        ui_label(slider_row, "", 12, BLUE, True, SIDEBAR,
                 textvariable=self.conf_label_var).pack(side="right", padx=(5, 0))

        self.start_button = ui_button(settings, "▶   BẮT ĐẦU NHẬN DIỆN", self._start, bg=BLUE, pady=13)
        self.start_button.pack(fill="x", padx=19, pady=(5, 10))
        self.stop_button = ui_button(settings, "■   DỪNG XỬ LÝ", self._stop, bg=PANEL_ALT, fg=TEXT, pady=12)
        self.stop_button.pack(fill="x", padx=19)
        self.stop_button.config(state="disabled", fg=MUTED)
        ui_button(settings, "Mở thư mục kết quả ↗", self._open_outputs,
                  bg=SIDEBAR, fg=GREEN).pack(fill="x", padx=19, pady=(18, 0))
        ui_label(settings, "Kết quả lưu trong outputs/\nLog CSV lưu trong outputs/logs/", 9, MUTED,
                 bg=SIDEBAR, justify="left").pack(side="bottom", anchor="w", padx=19, pady=20)

        # Center preview and indicators
        center = tk.Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True, padx=18, pady=18)
        self.center_area = center
        preview_top = tk.Frame(center, bg=BG)
        preview_top.pack(fill="x", pady=(0, 10))
        ui_label(preview_top, "MÀN HÌNH NHẬN DIỆN", 11, TEXT, True, BG).pack(side="left")
        self.small_status = ui_label(preview_top, "Chưa chạy", 9, MUTED, bg=BG)
        self.small_status.pack(side="right")
        preview_holder = tk.Frame(center, bg=PANEL, highlightbackground=BORDER,
                                  highlightthickness=1)
        preview_holder.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(preview_holder, bg="#090F1D", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Configure>", lambda *_: self._render_frame())
        ui_label(center, "", 9, MUTED, bg=BG, textvariable=self.frame_info_var,
                 anchor="w").pack(fill="x", pady=(7, 9))

        statrow = tk.Frame(center, bg=BG)
        statrow.pack(fill="x", pady=(0, 10))
        for title, variable, color in [
            ("ĐỘI MŨ", self.count_helmet_var, GREEN),
            ("KHÔNG ĐỘI MŨ", self.count_nohelmet_var, RED),
            ("FPS XỬ LÝ", self.count_fps_var, BLUE),
            ("ẢNH / FRAME", self.count_processed_var, YELLOW),
        ]:
            box = tk.Frame(statrow, bg=PANEL, padx=9, pady=9)
            box.pack(side="left", fill="both", expand=True, padx=(0, 7))
            ui_label(box, title, 8, MUTED, True, PANEL).pack(anchor="w")
            ui_label(box, "", 17, color, True, PANEL, textvariable=variable).pack(anchor="w", pady=(3, 0))
        logtitle = tk.Frame(center, bg=BG)
        logtitle.pack(fill="x", pady=(4, 5))
        ui_label(logtitle, "NHẬT KÝ HOẠT ĐỘNG", 9, MUTED, True, BG).pack(side="left")
        self.log_widget = tk.Text(center, bg=PANEL, fg=MUTED, height=5,
                                  font=("Consolas", 9), relief="flat", bd=0,
                                  padx=12, pady=9, wrap="word", state="disabled")
        self.log_widget.pack(fill="x")
        self._log("Chọn ảnh/video hoặc mở webcam để bắt đầu.")
        self._render_frame()

    def _entry(self, parent, variable):
        return tk.Entry(parent, textvariable=variable, bg=PANEL, fg=TEXT,
                        insertbackground=TEXT, selectbackground=BLUE_DARK,
                        relief="flat", bd=0, font=(FONT, 10), highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=BLUE)

    def _select_mode(self, mode):
        if self.is_running:
            return
        if self.mode in self.source_by_mode:
            self.source_by_mode[self.mode] = self.source_var.get()
        self.mode = mode
        if mode in self.source_by_mode:
            self.source_var.set(self.source_by_mode[mode])
        for key, button in self.mode_buttons.items():
            selected = key == mode
            button.config(bg=PANEL_ALT if selected else SIDEBAR,
                          fg=TEXT if selected else MUTED)
        if mode == "camera":
            self.source_block.pack_forget()
            self.camera_block.pack(fill="x", padx=19, pady=(7, 10), before=self.conf_heading)
        else:
            self.camera_block.pack_forget()
            self.source_block.pack(fill="x", padx=19, pady=(7, 10), before=self.conf_heading)
            self.input_title_var.set("Ảnh hoặc thư mục ảnh" if mode == "image" else "Tệp video")
            self.browse_folder_button.pack_forget()
            if mode == "image":
                self.browse_folder_button.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.small_status.config(text={"image": "Chế độ ảnh", "video": "Chế độ video",
                                       "camera": "Chế độ camera"}[mode])
        self.frame_info_var.set("Chưa có kết quả; chọn nguồn và nhấn Bắt đầu.")
        self.frame_data = None
        self._render_frame()

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Chọn model YOLO11s đã train",
                                          filetypes=[("YOLO model", "*.pt"), ("Tất cả", "*.*")])
        if path:
            self.model_var.set(path)

    def _browse_source(self):
        if self.mode == "image":
            types = [("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")]
        else:
            types = [("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Tất cả", "*.*")]
        path = filedialog.askopenfilename(title="Chọn dữ liệu kiểm thử", filetypes=types)
        if path:
            self.source_var.set(path)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if path:
            self.source_var.set(path)

    def _update_model_status(self):
        exists = resolve_path(self.model_var.get()).is_file()
        self.model_info_var.set("● Đã tìm thấy model" if exists else "● Chưa có model – chọn file best.pt đã train")

    def _log(self, message):
        self.log_widget.config(state="normal")
        self.log_widget.insert("end", "• " + str(message) + "\n")
        self.log_widget.see("end")
        self.log_widget.config(state="disabled")

    def _set_running(self, active):
        self.is_running = active
        self.start_button.config(state="disabled" if active else "normal",
                                 fg=MUTED if active else TEXT)
        self.stop_button.config(state="normal" if active else "disabled",
                                fg=TEXT if active else MUTED)
        self.model_entry.config(state="disabled" if active else "normal")
        self.source_entry.config(state="disabled" if active else "normal")
        self.camera_entry.config(state="disabled" if active else "normal")
        self.conf_scale.config(state="disabled" if active else "normal")
        self.header_status.config(text="●  ĐANG XỬ LÝ" if active else "●  SẴN SÀNG",
                                  fg=YELLOW if active else GREEN)

    def _start(self):
        model = resolve_path(self.model_var.get())
        if not model.is_file():
            messagebox.showwarning("Thiếu model", "Chưa có file best.pt/best_v1.pt.\n"
                                   "Nhấn nút ... cạnh MODEL để chọn file đã train.")
            return
        source = ""
        camera_id = 0
        if self.mode != "camera":
            source = str(resolve_path(self.source_var.get()))
            if not Path(source).exists():
                messagebox.showwarning("Thiếu đầu vào", f"Không tìm thấy ảnh/video:\n{source}")
                return
            if self.mode == "video" and not Path(source).is_file():
                messagebox.showwarning("Video không hợp lệ", "Hãy chọn một tệp video cụ thể.")
                return
        else:
            try:
                camera_id = int(self.camera_var.get())
                if camera_id < 0:
                    raise ValueError()
            except ValueError:
                messagebox.showwarning("Camera ID", "ID camera phải là số nguyên không âm, thường là 0.")
                return
        cfg = GuiConfig(mode=self.mode, model=str(model), source=source,
                        confidence=self.conf_var.get(), camera_id=camera_id,
                        record_camera=self.record_var.get(),
                        output_root=str(BASE / "outputs"))
        self.stop_event.clear()
        with self.frame_lock:
            self.latest_frame = None
        self.count_helmet_var.set("0")
        self.count_nohelmet_var.set("0")
        self.count_fps_var.set("—")
        self.count_processed_var.set("0")
        self._set_running(True)
        self._log(f"Bắt đầu {self.mode}; confidence={cfg.confidence:.2f}")
        self.worker = Thread(target=self._worker_run, args=(cfg,), daemon=True)
        self.worker.start()

    def _worker_run(self, cfg):
        def on_frame(frame, counts, fps, index, total, source):
            with self.frame_lock:
                self.latest_frame = (frame, counts, fps, index, total, source)

        def on_message(message):
            self.events.put(("log", message))
        try:
            stats, csv_path = process(cfg, self.stop_event, on_frame, on_message)
            self.events.put(("done", stats, str(csv_path)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll(self):
        if self.closing:
            return
        with self.frame_lock:
            packet = self.latest_frame
            self.latest_frame = None
        if packet is not None:
            frame, counts, fps, index, total, name = packet
            self.frame_data = frame
            self._render_frame()
            self.count_helmet_var.set(str(counts["helmet"]))
            self.count_nohelmet_var.set(str(counts["no_helmet"]))
            self.count_fps_var.set(f"{fps:.1f}")
            self.count_processed_var.set(f"{index}/{total}" if total else str(index))
            self.frame_info_var.set(f"{name}  ·  Helmet: {counts['helmet']}  ·  No Helmet: {counts['no_helmet']}")
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "log":
                    self._log(event[1])
                elif event[0] == "done":
                    stats, csv_path = event[1:]
                    self._set_running(False)
                    self._log(f"Hoàn tất: {stats.frames} ảnh/frame | {stats.elapsed_seconds:.1f} giây")
                    self._log(f"Kết quả: {stats.output}")
                    self._log(f"CSV: {csv_path}")
                    self.small_status.config(text="Đã hoàn tất")
                elif event[0] == "error":
                    self._set_running(False)
                    self._log("LỖI: " + event[1])
                    self.small_status.config(text="Có lỗi")
                    messagebox.showerror("Không thể chạy nhận diện", event[1])
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _render_frame(self):
        if not hasattr(self, "canvas"):
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        self.canvas.delete("all")
        if self.frame_data is None:
            self.canvas.create_text(cw / 2, ch / 2 - 25, text="◎", fill=BLUE,
                                    font=(FONT, 52, "bold"))
            self.canvas.create_text(cw / 2, ch / 2 + 32, text="KHUNG HÌNH NHẬN DIỆN",
                                    fill=TEXT, font=(FONT, 14, "bold"))
            self.canvas.create_text(cw / 2, ch / 2 + 58,
                                    text="Kết quả phát hiện sẽ hiển thị trực tiếp tại đây",
                                    fill=MUTED, font=(FONT, 10))
            return
        import cv2
        rgb = cv2.cvtColor(self.frame_data, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((max(1, cw - 16), max(1, ch - 16)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image, master=self)
        self.canvas.create_image(cw / 2, ch / 2, image=self.photo, anchor="center")

    def _stop(self):
        if self.is_running:
            self.stop_event.set()
            self._log("Đang yêu cầu dừng sau frame hiện tại...")
            self.small_status.config(text="Đang dừng")
            self.stop_button.config(state="disabled")

    def _open_outputs(self):
        output = BASE / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(output))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output)])
            else:
                subprocess.Popen(["xdg-open", str(output)])
        except (OSError, AttributeError) as exc:
            messagebox.showinfo("Thư mục kết quả", f"Bạn có thể mở thủ công:\n{output}\n\n{exc}")

    def _on_close(self):
        if self.is_running:
            if not messagebox.askyesno("Thoát chương trình", "Chương trình đang nhận diện.\nDừng và thoát?"):
                return
            self.stop_event.set()
            self._wait_close()
        else:
            self.closing = True
            self.destroy()

    def _wait_close(self):
        if self.worker is not None and self.worker.is_alive():
            self.after(100, self._wait_close)
        else:
            self.closing = True
            self.destroy()


def main():
    app = HelmetApp()
    app.mainloop()


if __name__ == "__main__":
    main()
