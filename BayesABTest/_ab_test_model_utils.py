"""Utils files for ab_test_model."""
# pylint: disable=no-member


class _ab_test_utils:
    def _format_axis_as_percent(self, locs, labels):
        labels = []
        for i in range(len(locs)):
            labels.append('{:.0%}'.format(locs[i]))
        return labels

    def _stringify_variants(self, variants=[]):
        if variants == []:
            strings = [self.control_bucket_name] + self.variant_bucket_names
        else:
            strings = variants
        last_word = []
        if len(strings) == 1:
            title = strings[0]
        elif len(strings) == 2:
            title = strings[0] + ' and ' + strings[1]
        else:
            last_word = strings[-1]
            strings[-1] = 'and '
            title = ', '.join(strings)
            title += str(last_word)
        return title

    def _format_title(self, string):
        length = len(string)
        if length < 40:
            return string
        newline = False
        for index, char in enumerate(string):
            if index < 40:
                continue
            if index % 40 == 0 and not newline:
                newline = True
                continue
            if newline and char == ' ':
                string = list(string)
                string[index] = '\n'
                string = ''.join(string)
                newline = False
        return string
