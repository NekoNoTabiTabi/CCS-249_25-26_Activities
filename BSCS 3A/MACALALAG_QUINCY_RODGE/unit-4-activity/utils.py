class Utils:

    def get_str(self, prompt):

        value = ""

        while value.strip() == "":
            value = input(prompt)

        return value


utils = Utils()
