Data Mining Algorithms (From Scratch)

Educational implementations of classic Data Mining algorithms using NumPy.

Hiện tại đã implement:

Gaussian Naive Bayes

Multinomial Naive Bayes

Bernoulli Naive Bayes

Installation

Khuyến nghị tạo virtual environment:

python -m venv .venv
.venv\Scripts\activate (Windows)

source .venv/bin/activate (macOS/Linux)

Cài dependencies:

pip install numpy scikit-learn matplotlib pytest

How to Run Tests

Chạy một test cụ thể (khuyến nghị):

py -m tests.test_gaussian
py -m tests.test_multinomial
py -m tests.test_bernoulli

Hoặc chạy toàn bộ:

pytest -s

Lưu ý: Không chạy trực tiếp file test kiểu
py tests/test_gaussian.py

Hãy dùng -m hoặc pytest để Python resolve module đúng cách.