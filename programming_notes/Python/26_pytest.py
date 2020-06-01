# Adding messages to assert statements
import pytest
import convert_string_to_int
def test_on_string_with_one_comma():
    test_argument = "2,081"
    expected = 2081
    actual = convert_string_to_int(test_argument)
    message = "convert_string_to_int('2,081') should return the int 2081, but it actually returned {0}".format(actual)
    assert actual == expected, message

# Testing float return values
# Due to floating-point arithmetic, 0.1 + 0.1 + 0.1 == 0.3 returns False
actual = 0.3
expected = 0.1 + 0.1 + 0.1
assert actual == pytest.approx(expected), message

# Testing for exceptions
with pytest.raises(ValueError) as exc_info:
    raise ValueError("Silence me!")
expected_error_message = "Silence me!"
assert exc_info.match(expected_error_message)

# Stop after 1st failed tests
pytest -x

# Run tests with names matching a pattern
pytest -k "pattern"

# Expecting tests to fail
class TestFunction(object):
	@pytest.mark.xfail(reason="why it should fail?")
	def test_something(self):
		...
# Showing reason for failing
pytest -rx

# Skip test conditianally (e.g. if Pythion version is higher than 2.7)
class TestFunction(object):
	@pytest.mark.skipif(sys.version_info > (2, 7), reason="requires Python <= 2.7")
	def test_something(self):
		...
# Show reason for skipping in the report
pytest -rs
# Show reasons for both xfailed and skipped tests
pytests -rsx

# Tesing functions that read or save data
# - Test the get_data_as_numpy_array() function takes the path to clean data file as the first 
#   argument and the number of columns of data as the second argument.
# - Using the fixture clean_data_file(), which:
#       * creates a clean data file in the setup,
#       * yields the path to the clean data file,
#       * removes the clean data file in the teardown.
# Add a decorator to make this function a fixture
@pytest.fixture
def clean_data_file():
    file_path = "clean_data_file.txt"
    with open(file_path, "w") as f:
        f.write("201\t305671\n7892\t298140\n501\t738293\n")
    yield file_path
    os.remove(file_path)
    
def test_on_clean_file(clean_data_file):
    expected = np.array([[201.0, 305671.0], [7892.0, 298140.0], [501.0, 738293.0]])
    actual = get_data_as_numpy_array(clean_data_file, 2)
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual) 

# Even better: using tmpdir fixture which takes care of deleting files
# (example fixture for creating empty file)
@pytest.fixture
def empty_file(tmpdir):
    file_path = tmpdir.join("empty.txt")
    open(file_path, "w").close()
    yield file_path


# Mocking -------------------------------------------------------------------------------------------
import pytest-mock
import unittest.mock.call

def add_one(x):
	return x + 2

def subtract_two(x):
	return x - 2

def add_and_subtract(x):
	x = add_one(x)
	x = subtract_two(x)
	return x

# Example: add_one() has a bug, but add_and_subtract() is correct. We want the test for it to pass,
# regardless of the bugs in its dependency.
# 1. Program a bug-free dependency
def add_one_bug_free(num):
	return_values = {
		0: 1,
		1: 2,
		2: 3
	}
	return return_values[num]
# 2. Mock a dependency
# Mocking helps us replace a dependency with a MagicMock() object. Usually, the MagicMock() 
# is programmed to be a bug-free version of the dependency. To verify whether the function under 
# test works properly with the dependency, you simply check whether the MagicMock() is called with 
# the correct arguments and in the right order.
def test_adding_and_subtracting(self, input_num, mocker):
	add_one_mock = mocker.patch("project.utils.add_one", side_effect=add_one_bug_free)
	res = add_and_subtract(input_num)
	# check if add_and_subtract() called the dependency coorectly
	assert add_one_mock.call_args_list == [call(0), call(1), call(2)]
	assert # stugg checking if res is as expected


# Testing models -------------------------------------------------------------------------------------
def train_linreg_model(training_set):
	# ...
	return intercept, slope

def test_on_linear_data():
	# y = 2x + 1
	test_argument = np.array([[1, 3],
							  [2, 5],
							   3, 7])
	expected_slope = 2
	expected_intercept = 1
	intercept, slope = train_linreg_model(test_argument)
	assert slope == pytest.approx(expected_slope)


# Testing plots --------------------------------------------------------------------------------------
# Plots have to many attributes to compare them. Instead do:
# 1. Baseline generation: convert plot with specific arguments to png
# 2. Testing: create pmg image with test arguments and compare images with pytest-mpl
# It can all be done at once using a decorator to the testing function.

# Test function get_plot_for_best_fit_line():
# Test class
class TestGetPlotForBestFitLine(object):
    @pytest.mark.mpl_image_compare
    def test_plot_for_almost_linear_data(self):
        slope = 5.0
        intercept = -2.0
        x_array = np.array([1.0, 2.0, 3.0])
        y_array = np.array([3.0, 8.0, 11.0])
        title = "Test plot for almost linear data"
        return get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title)

# Create baseline image
pytest -k "test_plot_line" --mpl-generate-path project/visualizations/baseline
# Run test
pytest -k "test_plot_line" --mpl

# In case of failures (test image different from baseline), pytest will save the baseline images, 
# actual images, and images containing the pixelwise difference in a temporary folder.





