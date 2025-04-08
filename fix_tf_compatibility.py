import os

print("Creating TensorFlow 2.x compatibility fixes...")

# First, add TF compatibility mode to main files
files_to_modify = ['main.py', 'model.py', 'ops.py']

for file_path in files_to_modify:
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Add compatibility mode if not already present
        if not "tf.compat.v1.disable_eager_execution()" in content:
            # Find suitable insertion point
            if "import tensorflow as tf" in content:
                content = content.replace("import tensorflow as tf", 
                                         "import tensorflow as tf\ntf.compat.v1.disable_eager_execution()")
            else:
                content = "import tensorflow as tf\ntf.compat.v1.disable_eager_execution()\n\n" + content
        
        # Replace common TF 1.x APIs with TF 2.x compatible versions
        replacements = [
            ("tf.app", "tf.compat.v1.app"),
            ("tf.placeholder", "tf.compat.v1.placeholder"),
            ("tf.Session", "tf.compat.v1.Session"),
            ("tf.ConfigProto", "tf.compat.v1.ConfigProto"),
            ("tf.get_variable", "tf.compat.v1.get_variable"),
            ("tf.variable_scope", "tf.compat.v1.variable_scope"),
            ("tf.train.Saver", "tf.compat.v1.train.Saver"),
            ("tf.trainable_variables", "tf.compat.v1.trainable_variables"),
            ("tf.global_variables_initializer", "tf.compat.v1.global_variables_initializer"),
            ("tf.train.AdamOptimizer", "tf.compat.v1.train.AdamOptimizer"),
            ("tf.train.get_checkpoint_state", "tf.compat.v1.train.get_checkpoint_state"),
            ("tf.random_normal", "tf.random.normal"),
            ("tf.truncated_normal", "tf.random.truncated_normal"),
            ("tf.summary.", "tf.compat.v1.summary."),
            ("tf.nn.conv2d_transpose", "tf.compat.v1.nn.conv2d_transpose"),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        with open(file_path, 'w') as file:
            file.write(content)
        
        print(f"Updated {file_path} with TF 2.x compatibility fixes")

# Fix the slim issue in utils.py
if os.path.exists("utils.py"):
    with open("utils.py", 'r') as file:
        content = file.read()
    
    # Replace the slim import
    if "import tensorflow.contrib.slim as slim" in content:
        slim_replacement = """
# TensorFlow 2.x compatibility - slim replacement
import tensorflow as tf

# Simple replacement for slim functionality we need
class SlimReplacement:
    class model_analyzer:
        @staticmethod
        def analyze_vars(variables, print_info=True):
            if print_info:
                for var in variables:
                    print(f"{var.name} - {var.shape}")

slim = SlimReplacement()
"""
        content = content.replace("import tensorflow.contrib.slim as slim", slim_replacement)
    
    # Replace show_all_variables function
    show_all_vars_replacement = """
def show_all_variables():
    model_vars = tf.compat.v1.trainable_variables()
    print("Model Variables:")
    for var in model_vars:
        print(f"  {var.name} - {var.shape}")
"""
    
    if "def show_all_variables():" in content:
        # Find the function and replace it
        start_idx = content.find("def show_all_variables():")
        end_idx = content.find("\n\n", start_idx)
        if end_idx == -1:  # In case there's no double newline after function
            end_idx = content.find("\ndef", start_idx)
        
        if end_idx != -1:
            content = content[:start_idx] + show_all_vars_replacement + content[end_idx:]
    
    with open("utils.py", 'w') as file:
        file.write(content)
    
    print("Updated utils.py with slim compatibility replacement")

# Fix the Covtype dataset preprocessing if needed
if os.path.exists("preprocess_covtype.py"):
    with open("preprocess_covtype.py", 'r') as file:
        content = file.read()
    
    # Update to handle potential column name issues
    if "covtype_df['label'] = (covtype_df['Cover_Type'] > 3).astype(int)" in content:
        modified_content = content.replace(
            "covtype_df['label'] = (covtype_df['Cover_Type'] > 3).astype(int)",
            "# Handle potential column name variations\n"
            "target_col = 'Cover_Type' if 'Cover_Type' in covtype_df.columns else covtype_df.columns[-1]\n"
            "covtype_df['label'] = (covtype_df[target_col] > 3).astype(int)"
        )
        
        modified_content = modified_content.replace(
            "covtype_df = covtype_df.drop('Cover_Type', axis=1)",
            "covtype_df = covtype_df.drop(target_col, axis=1)"
        )
        
        with open("preprocess_covtype.py", 'w') as file:
            file.write(modified_content)
        
        print("Updated preprocess_covtype.py to handle column name variations")

print("Compatibility fixes completed!") 