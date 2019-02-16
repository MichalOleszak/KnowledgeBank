# Check if a funtion is an S3 generic or method -------------------------------
library(pryr)
is_s3_generic("t")           # generic transpose function
is_s3_method("t.data.frame") # transpose method for data.frames
is_s3_method("t.test")       # a function for Student's t-tests


# Creating a generic function with methods ------------------------------------
get_n_elements <- function(x, ...) {
  UseMethod("get_n_elements")
}

# Create a data.frame method for get_n_elements
get_n_elements.data.frame <- function(x, ...) {
  nrow(x) * ncol(x)
}

# Create a default method for get_n_elements
get_n_elements.default <- function(x, ...) {
  length(unlist(x))
}


# Find available methods ------------------------------------------------------
# for a generic function
methods(print)
# for a class
methods(class = data.frame)     # both S3 and S4 methods
.S3methods(class = data.frame)  # onlt S3 methods


# Primitive Functions (written in C) ------------------------------------------
# Primitive functions include language elements, like if and for, 
# operators like + and $, mathematical functions like exp and sin, 
# and also some S3 generics:
.S3PrimitiveGenerics 
# In contrast to regular generics, primitive generics don't throw an error when 
# no method is found, as they go to the C code to determine the type using typeof()
# Consequently, overwriting the class does not break behaviour.


# Test for arbitrary classes --------------------------------------------------
x <- c("a", "e", "i", "o", "u")
class(x) <- c("vowels", "letters", "character")
inherits(x, "vowels")


# Methods for multiple classes ------------------------------------------------
kitty <- "meow!"
class(kitty) <- c("cat", "mammal", "character")
what_am_i <- function(x, ...) {
  UseMethod("what_am_i")
}
# cat method
what_am_i.cat <- function(x, ...) {
  message("I'm a cat")
  NextMethod(what_am_i)
}
# mammal method
what_am_i.mammal <- function(x, ...) {
  message("I'm a mammal")
  NextMethod(what_am_i)
}
# character method
what_am_i.character <- function(x, ...) {
  message("I'm a character vector")
}

what_am_i(kitty)


# Working with R6 -------------------------------------------------------------
library(R6)
library(assertive.numbers)
library(assertive.types)
# Create class generator (a.k.a. factory)
# - user interface is stored on public
# - internal data is in private (names start with .. for standing out)
# - private fields can be accessed using private$
# - public fields in self can be accessed using self$
# - public fields in parent can be accessed using super$
# - initialize() allows to set private fields when objects are created
# - Active Bindings stored in active allow for controlled private access:
#     -> To create an active binding to get a private data field (i.e. a "read-only" binding), 
#        you create a function with no arguments that simply returns the private element
#     -> Active bindings can also be used to set private fields. In this case, the binding 
#        function should accept a single argument, named "value".
microwave_oven_factory <- R6Class(
  "MicrowaveOven",
  private = list(
    ..power_rating_watts = 800,
    ..door_is_open = FALSE
  ),
  public = list(
    cook = function(time_seconds) {
      Sys.sleep(time_seconds)
      print("Your food is cooked!")
    },
    open_door = function() {
      private$..door_is_open = TRUE
    },
    close_door = function() {
      private$..door_is_open = FALSE
    },
    initialize = function(power_rating_watts, door_is_open) {
      if (!missing(power_rating_watts)) {
        private$..power_rating_watts <- power_rating_watts
      }
      if (!missing(door_is_open)) {
        private$..door_is_open <- door_is_open
      }
    }
  ),
  active = list(
    power_rating_watts = function(value) {
      if(missing(value)) {
        private$..power_rating_watts
      } else {
        assert_is_a_number(value)
        assert_all_are_in_closed_range(
          value, lower = 0, upper = private$..power_rating_watts
        )
        private$..power_rating_watts <- value
      }
    }
  )
)

# Make a microwave
microwave_oven <- microwave_oven_factory$new(power_rating_watts = 650,
                                             door_is_open = TRUE)
# Use its methods
microwave_oven$cook(5)
microwave_oven$close_door()

# Make use of Active Bindings to get and set data
microwave_oven$power_rating_watts
microwave_oven$power_rating_watts <- "400"
microwave_oven$power_rating_watts <- 1600
microwave_oven$power_rating_watts <- 400
microwave_oven$power_rating_watts


# Propagating Functionality with Inheritance ----------------------------------
# Child inherits everything from the parent
#  - extend functionality by adding cook baked potato method
#  - overwrite parent's cook method
fancy_microwave_oven_factory <- R6Class(
  "FancyMicrowaveOven",
  inherit = microwave_oven_factory,
  public = list(
    cook_baked_potato = function() {
      self$cook(3)
    },
    cook = function(time_seconds) {
      super$cook(time_seconds)
      message("Enjoy your dinner!")
    }
  )
)
fancy_microwave <- fancy_microwave_oven_factory$new()
fancy_microwave$power_rating_watts
fancy_microwave$cook(1)
fancy_microwave$cook_baked_potato()
fancy_microwave$cook(1)


# Multiple levels of inheritance ----------------------------------------------
# By default, child can only inherit from its parent, not grandparant;
# Intermediate class (the parent) can expose its parent using an active binding
# named super_ that returns super
fancy_microwave_oven_factory <- R6Class(
  "FancyMicrowaveOven",
  inherit = microwave_oven_factory,
  public = list(
    cook_baked_potato = function() {
      self$cook(3)
    },
    cook = function(time_seconds) {
      super$cook(time_seconds)
      message("Enjoy your dinner!")
    }
  ),
  active = list(
    super_ = function() super
  )
)

high_end_microwave_oven_factory <- R6Class(
  "HighEndMicrowaveOven",
  inherit = fancy_microwave_oven_factory,
  public = list(
    cook = function(time_seconds) {
      super$super_$cook(time_seconds)
      message("PIZZA")
    }
  )
)

high_end_microwave <- high_end_microwave_oven_factory$new()
high_end_microwave$cook(1)


# Environments ----------------------------------------------------------------
# Environments use copy by reference, so that all copies contain the same values:
lst <- list(
  perfect = c(6, 28, 496),
  bases = c("A", "C", "G", "T")
)
env <- list2env(lst)
lst2 <- lst
lst$bases <- c("A", "C", "G", "U")
identical(lst$bases, lst2$bases)
env2 <- env
env$bases <- c("A", "C", "G", "U")
identical(env$bases, env2$bases)

# R6 classes can use environments' copy by reference behavior to share fields between objects:
# Complete the class definition
microwave_oven_factory2 <- R6Class(
  "MicrowaveOven",
  private = list(
    shared = {
      e = new.env()
      e$safety_warning <- "Warning. Do not try to cook metal objects."
      e
    }
  ),
  active = list(
    safety_warning = function(value) {
      if (missing(value)) {
        private$shared$safety_warning
      } else {
        private$shared$safety_warning <- value
      }
    }
  )
)
# Create two microwave ovens
a_microwave_oven <- microwave_oven_factory2$new()
another_microwave_oven <- microwave_oven_factory2$new()
# Change the safety warning for a_microwave_oven
a_microwave_oven$safety_warning <- "Warning. If the food is too hot you may scald yourself."
# Verify that the warning has change for another_microwave
another_microwave_oven$safety_warning


# Cloning objects -------------------------------------------------------------
# Clone() copies by value: while changing filed in the original, the clone is unaffected
a_microwave_oven <- microwave_oven_factory$new()
assigned_microwave_oven <- a_microwave_oven
cloned_microwave_oven <- a_microwave_oven$clone()

a_microwave_oven$power_rating_watts <- 400
identical(a_microwave_oven$power_rating_watts, assigned_microwave_oven$power_rating_watts)
identical(a_microwave_oven$power_rating_watts, cloned_microwave_oven$power_rating_watts)  

# If an R6 object contains another R6 object in one or more of its fields, then by default 
# clone() will copy the R6 fields by reference. To copy those R6 fields by value, the clone() 
# method must be called with the argument deep = TRUE.


# Finalising ------------------------------------------------------------------
# Just as an R6 class can define a public initialize() method to run custom code when objects 
# are created, they can also define a public finalize() method to run custom code when objects 
# are destroyed. finalize() should take no arguments. It is typically used to close connections 
# to databases or files, or undo side-effects such as changing global options() or graphics 
# par()ameters. The finalize() method is called when the object is removed from memory by R's 
# automated garbage collector. You can force a garbage collection by typing gc().
smart_microwave_oven_factory <- R6Class(
  "SmartMicrowaveOven",
  inherit = microwave_oven_factory, # Specify inheritance
  private = list(
    conn = NULL
  ),
  public = list(
    initialize = function() {
      private$conn = dbConnect(SQLite(), "cooking-times.sqlite")
    },
    get_cooking_time = function(food) {
      dbGetQuery(
        private$conn,
        sprintf("SELECT time_seconds FROM cooking_times WHERE food = '%s'", food)
      )
    },
    finalize = function() {
      # Print a message
      message("Disconnecting from the cooking times database.")
      # Disconnect from the database
      dbDisconnect(private$conn)
    }
  )
)
