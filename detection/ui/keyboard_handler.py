"""
Keyboard Handler - Manages keyboard input with action callbacks.

SINGLE RESPONSIBILITY: Handle keyboard input and dispatch actions.

CALLBACK PATTERN: Register functions to be called when keys are pressed.
"""

import cv2
from typing import Dict, Callable


class KeyboardHandler:
    """
    Handles keyboard input using callback pattern.

    DESIGN PATTERN: Command Pattern / Observer Pattern
    - Register actions for specific keys
    - When key pressed, execute registered action

    KEYWORD EXPLANATION:
        Callable: Type hint for function (like function pointer in C++)
        Dict[str, Callable]: Dictionary mapping keys to functions
    """

    def __init__(self):
        """
        Initialize keyboard handler.

        CALLBACK STORAGE:
            key_actions: Dict mapping key characters to callback functions
            Like: std::map<char, std::function<void()>> in C++
        """
        # Storage for key → action mappings
        # Dict[str, Callable]: Dictionary where key=string, value=function
        self.key_actions: Dict[str, Callable] = {}

    def register(self, key: str, action: Callable, description: str = ""):
        """
        Register a callback function for a key press.

        CALLBACK REGISTRATION:
            When 'key' is pressed, 'action' function will be called

        Args:
            key: Key character (e.g., 'q', 's', 'r')
            action: Function to call when key is pressed
            description: Human-readable description of what this does

        Example:
            def quit_app():
                print("Quitting!")
                sys.exit()

            handler.register('q', quit_app, "Quit application")
        """
        self.key_actions[key] = action
        if description:
            print(f"  '{key}' → {description}")

    def handle(self, wait_ms: int = 1) -> bool:
        """
        Check for keyboard input and execute registered actions.

        Call this in your main loop!

        Args:
            wait_ms: Milliseconds to wait for key press

        Returns:
            True if a key was handled, False otherwise

        PYTHON KEYWORDS:
            cv2.waitKey(wait_ms): OpenCV function to wait for keyboard
            & 0xFF: Bitmask to get only ASCII value
            ord('q'): Converts character 'q' to ASCII code
        """
        # Wait for key press (OpenCV function)
        key = cv2.waitKey(wait_ms) & 0xFF

        # If no key pressed, return
        if key == 255:  # No key
            return False

        # Convert key code to character
        key_char = chr(key)

        # Execute action if registered
        if key_char in self.key_actions:
            self.key_actions[key_char]()  # Call the function!
            return True

        return False

    def print_help(self):
        """Print all registered key bindings."""
        print("\n" + "=" * 60)
        print("Keyboard Controls:")
        print("=" * 60)
        for key, action in self.key_actions.items():
            # Get function name for display
            func_name = action.__name__ if hasattr(action, '__name__') else str(action)
            print(f"  '{key}' → {func_name}")
        print("=" * 60 + "\n")


# =============================================================================
# DESIGN PATTERN: CALLBACK PATTERN
# =============================================================================
"""
The Callback Pattern lets you register functions to be called later.

C++ ANALOGY:
    // C++ with function pointers
    class KeyboardHandler {
        std::map<char, std::function<void()>> actions;

        void register(char key, std::function<void()> func) {
            actions[key] = func;
        }

        void handle() {
            char key = getKey();
            if (actions.find(key) != actions.end()) {
                actions[key]();  // Call the function!
            }
        }
    };

PYTHON VERSION (this class):
    class KeyboardHandler:
        def __init__(self):
            self.key_actions = {}  # Dict instead of map

        def register(self, key, action):
            self.key_actions[key] = action

        def handle(self):
            key = cv2.waitKey(1)
            if key in self.key_actions:
                self.key_actions[key]()  # Call it!

USAGE EXAMPLE:
    handler = KeyboardHandler()

    # Register actions
    handler.register('q', lambda: sys.exit(), "Quit")
    handler.register('s', toggle_autopilot, "Toggle autopilot")

    # In main loop
    while True:
        # ... process frame ...
        handler.handle()  # Check for keys and execute actions

BENEFITS:
    - Separates key handling from main logic
    - Easy to add/remove key bindings
    - Testable (can mock key presses)
    - Clean code organization
"""


if __name__ == "__main__":
    # Example usage
    print("Testing KeyboardHandler...")

    # Create handler
    handler = KeyboardHandler()

    # Register some actions
    def quit_action():
        print("Quit action called!")

    def toggle_action():
        print("Toggle action called!")

    handler.register('q', quit_action, "Quit application")
    handler.register('t', toggle_action, "Toggle mode")

    # Print help
    handler.print_help()

    print("Keyboard handler ready to use!")
