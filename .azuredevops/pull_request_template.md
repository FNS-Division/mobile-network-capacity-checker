## 📝 Pull Request Checklist

**Thank you for your contribution!** Before submitting your pull request, ensure you've diligently completed the tasks listed below. This process helps us maintain high code quality and facilitates the review process.

### 🧪 Testing
- [] **Unit Tests**
  - Verify that all unit tests pass.
  - Execute the tests from the project's root directory:
    ```bash
    pytest
    ```

### 🎨 Code Quality
- [] **Flake8 Compliance**
  - Confirm that your code adheres to Flake8 standards.
  - Perform the checks from the project's root directory:
    ```bash
    flake8 .
    ```
- **Pro Tip:** Use `autopep8` to automatically correct PEP 8 issues in your code:
    ```bash
    autopep8 --in-place --recursive .
    ```

### 📒 Version control for notebooks
- [] **Clear output from notebooks**
  - In JupyterLab: Use the Run menu -> Clear All Outputs.
- **Pro Tip:** Use `nbstripout ` to automatically remove output from Jupyter notebooks before committing:
    ```bash
    find . -name "*.ipynb" -exec nbstripout {} \;
    ```
- [] **Flag important notebook changes**
  - In your message below, explicitly mention which notebooks have been updated. Only commit Jupyter notebook files that have meaningful updates. Avoid committing notebooks that haven't been changed to reduce clutter.
  - Regularly review and remove notebooks that are no longer required. 

### ✅ Final Review
- [] **Label Your Pull Request**
  - Apply relevant labels to describe the nature of your changes (e.g., `bugfix`, `enhancement`, `docs`).
- [] **Link to Your DevOps Work Item**
  - Use the "Linked work items" field to attach the relevant Azure DevOps item if applicable
- [] **Assignees**
  - Assign the pull request to yourself and any reviewers you believe are necessary to evaluate your changes.

---

Adhering to these guidelines is crucial for a smooth and efficient review process. **We appreciate your efforts to contribute to the project's excellence!**