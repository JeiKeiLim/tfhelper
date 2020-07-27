import tensorflow as tf


def tfprint(x, len=None, format="{}, ", name="var"):
    if len is None:
        len = tf.shape(x)[0]

    tf.print(f"{name} (", len, "): ", sep="", end="")

    def print_var(i):
        # msg = tf.strings.format(format, x[i])
        tf.print(x[i], ",", end="", sep="")

        return i + 1

    tf.while_loop(lambda i: i < len, print_var, (0,))

    tf.print("")
