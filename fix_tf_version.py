# Create a file called fix_tf2_compatibility.py
import os

# Fix the utils.py file
with open('utils.py', 'r') as file:
    content = file.read()

# Replace tensorflow.contrib.slim imports
content = content.replace("import tensorflow.contrib.slim as slim", 
                         "# TF 2.x compatibility\nimport tensorflow as tf\n\n# Replace slim functionality\nclass SlimReplacement:\n    @staticmethod\n    def model_analyzer_analyze_vars(vars, print_info=True):\n        if print_info:\n            for v in vars:\n                print(f'{v.name}, {v.shape}')\n\nslim = SlimReplacement()")

# Fix show_all_variables function
content = content.replace("def show_all_variables():\n    model_vars = tf.trainable_variables()\n    slim.model_analyzer.analyze_vars(model_vars, print_info=True)",
                         "def show_all_variables():\n    model_vars = tf.compat.v1.trainable_variables()\n    for v in model_vars:\n        print(f'{v.name}, {v.shape}')")

with open('utils.py', 'w') as file:
    file.write(content)

# Also fix the invalid escape sequence in model.py if it exists
if os.path.exists('model.py'):
    with open('model.py', 'r') as file:
        content = file.read()

    # Fix re.finditer escape sequence
    if "(\\d+)(?!.*\\d)" in content:
        content = content.replace("(\\d+)(?!.*\\d)", r"(\d+)(?!.*\d)")

    # Add TF 1.x compatibility mode at the top
    if not "tf.compat.v1.disable_eager_execution()" in content:
        content = "import tensorflow as tf\ntf.compat.v1.disable_eager_execution()\n\n" + content

    # Replace tf.app with tf.compat.v1.app
    content = content.replace("tf.app", "tf.compat.v1.app")
    
    # Replace other common TF 1.x functions
    content = content.replace("tf.placeholder", "tf.compat.v1.placeholder")
    content = content.replace("tf.Session", "tf.compat.v1.Session")
    content = content.replace("tf.ConfigProto", "tf.compat.v1.ConfigProto")
    content = content.replace("tf.train.get_checkpoint_state", "tf.compat.v1.train.get_checkpoint_state")
    
    with open('model.py', 'w') as file:
        file.write(content)

# Fix main.py 
if os.path.exists('main.py'):
    with open('main.py', 'r') as file:
        content = file.read()
    
    # Add TF 1.x compatibility mode
    if not "tf.compat.v1.disable_eager_execution()" in content:
        insertion_point = content.find("import tensorflow as tf")
        if insertion_point != -1:
            content = content[:insertion_point+22] + "\ntf.compat.v1.disable_eager_execution()" + content[insertion_point+22:]
        else:
            content = "import tensorflow as tf\ntf.compat.v1.disable_eager_execution()\n\n" + content
    
    # Replace tf.app with tf.compat.v1.app
    content = content.replace("tf.app", "tf.compat.v1.app")
    
    with open('main.py', 'w') as file:
        file.write(content)

# Fix ops.py if it exists
if os.path.exists('ops.py'):
    with open('ops.py', 'r') as file:
        content = file.read()
    
    # Replace common TF 1.x functions
    content = content.replace("tf.variable_scope", "tf.compat.v1.variable_scope")
    content = content.replace("tf.get_variable", "tf.compat.v1.get_variable")
    content = content.replace("tf.random_normal", "tf.random.normal")
    content = content.replace("tf.truncated_normal", "tf.random.truncated_normal")
    
    with open('ops.py', 'w') as file:
        file.write(content)

print("Files updated for TensorFlow 2.x compatibility")