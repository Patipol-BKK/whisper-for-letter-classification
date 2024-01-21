import time
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import signal

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, working_dir, script_path):
        super().__init__()
        self.working_dir = working_dir
        self.script_path = script_path
        self.subprocess = None

    def on_modified(self, event):
        common_path = os.path.commonpath([self.working_dir, event.src_path])
        # Check if event is in specified dir isn't pycache
        if common_path == self.working_dir and not '__pycache__' in event.src_path:
            if self.subprocess is not None:
                # os.kill(self.subprocess.pid, signal.SIGTERM)
                self.subprocess.terminate()
                
            print('Restarting script...')
            self.subprocess = subprocess.Popen([sys.executable, self.script_path])

if __name__ == '__main__':
    working_dir = 'src'
    script_path = 'src\\dev.py'  # Replace with your script path
    event_handler = FileChangeHandler(working_dir, script_path)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()