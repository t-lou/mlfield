import gi

gi.require_version("Gtk", "3.0")
import re  # noqa: E402

from gi.repository import Gtk  # noqa: E402


def trim_code(raw: str) -> str:
    # Remove triple-quoted blocks (""" ... """ or ''' ... ''')
    raw = re.sub(r'"""[\s\S]*?"""', "", raw)
    raw = re.sub(r"'''[\s\S]*?'''", "", raw)

    lines = raw.splitlines()
    out = []

    for line in lines:
        # Preserve indentation
        indent_match = re.match(r"^(\s*)", line)
        indent = indent_match.group(1)
        content = line[len(indent) :]

        # Remove empty lines
        if content.strip() == "":
            continue

        # Remove Python comments
        if content.strip().startswith("#"):
            continue

        # Remove inline Python comments
        content = re.sub(r"#.*", "", content).rstrip()

        # NOTE: Commented-out as I only need this script for python for now.
        # # Remove C/JS style comments
        # if content.strip().startswith("//"):
        #     continue
        # content = re.sub(r"//.*", "", content).rstrip()

        # # Remove block comments /* ... */
        # content = re.sub(r"/\*.*?\*/", "", content).rstrip()

        # Skip if empty after cleaning
        if content.strip() == "":
            continue

        # Reassemble with original indentation
        out.append(indent + content)

    return "\n".join(out)


class CodeTrimWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Code Trimmer for AI")
        self.set_default_size(900, 600)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.add(hbox)

        # Left textview (raw code)
        self.raw_buffer = Gtk.TextBuffer()
        raw_view = Gtk.TextView(buffer=self.raw_buffer)
        raw_view.set_wrap_mode(Gtk.WrapMode.NONE)

        raw_scroll = Gtk.ScrolledWindow()
        raw_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        raw_scroll.add(raw_view)

        # Right textview (trimmed code)
        self.trim_buffer = Gtk.TextBuffer()
        trim_view = Gtk.TextView(buffer=self.trim_buffer)
        trim_view.set_wrap_mode(Gtk.WrapMode.NONE)

        trim_scroll = Gtk.ScrolledWindow()
        trim_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        trim_scroll.add(trim_view)

        # Button
        button = Gtk.Button(label="Trimm")
        button.connect("clicked", self.on_trim_clicked)

        # Layout
        hbox.pack_start(raw_scroll, True, True, 0)
        hbox.pack_start(trim_scroll, True, True, 0)
        hbox.pack_start(button, False, False, 0)

    def on_trim_clicked(self, widget):
        start = self.raw_buffer.get_start_iter()
        end = self.raw_buffer.get_end_iter()
        raw_text = self.raw_buffer.get_text(start, end, True)

        trimmed = trim_code(raw_text)
        self.trim_buffer.set_text(trimmed)


def main():
    win = CodeTrimWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
