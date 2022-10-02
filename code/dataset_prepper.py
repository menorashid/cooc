import numpy
import pandas


class DatasetPrepper:

    def __init__(self, config):
        self.fields = config.fields
        self.dataset = pandas.read_csv(config.path, delimiter=config.delimiter, engine='python')
        self.drop_rows(config)
        self.header_aus = numpy.unique(self.dataset[self.fields.code].values)

    def drop_rows(self, config):
        field_names = [self.fields.file, self.fields.subject, self.fields.code]
        fields_vals_to_drop = [
            config.files_to_exclude, config.subjects_to_exclude, config.codes_to_exclude
        ]
        for field_name, vals_to_drop in zip(field_names, fields_vals_to_drop):
            self.dataset = self.dataset[~self.dataset[field_name].isin(vals_to_drop)]

    def get_ows_dataset(self, ows):
        ows_df = []
        for file_name in numpy.unique(self.dataset[self.fields.file].values):
            ows_df.append(
                self._split_file_into_ows(
                    self.dataset.loc[self.dataset[self.fields.file] == file_name], ows
                )
            )
        return pandas.concat(ows_df)

    def _split_file_into_ows(self, dataset, ows):
        file_name = dataset[self.fields.file].iloc[0]
        min_start = numpy.min(dataset[self.fields.start].values)
        starts = dataset[self.fields.start].values - min_start
        stops = dataset[self.fields.stop].values - min_start
        aus = dataset[self.fields.code].values
        increment_starts = numpy.arange(0, numpy.max(stops)//ows * ows, ows)

        file_df = []

        for increment_start in increment_starts:
            increment_end = increment_start + ows
            increment_file_name = f'{file_name}_{increment_start}_{increment_end}'
            increment_pain_level = dataset[self.fields.pain_level].iloc[0]
            increment_subject = dataset[self.fields.subject].iloc[0]

            # get all the unique aus that were active during the increment
            to_include = numpy.logical_not(
                numpy.logical_or(starts < increment_start, stops >= increment_end)
            )
            aus_in_increment = numpy.unique(aus[to_include])

            # set their value to 1 for this file
            au_counts = numpy.in1d(self.header_aus, aus_in_increment).astype(numpy.int32)
            file_df.append(
                [increment_subject, increment_file_name, increment_pain_level] + list(au_counts)
            )

        headers = [self.fields.subject, self.fields.file, self.fields.pain_level] + \
                    list(self.header_aus)
        return pandas.DataFrame(file_df, columns=headers)
