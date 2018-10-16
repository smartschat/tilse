import datetime

from tilse.data import timelines


def post_process(ranked_sentences,
                 extents,
                 daily_summary_length,
                 timeline_length,
                 start,
                 end):
    """
    Post-process ranked sentences to obtain a timeline.

    In particular, greedily select sentences according to their ranks such that
    they respect constraints imposed by extents of dates, maximum daily
    summary length, maximum timeline length and start and end date.

    Params:
        ranked_sentences (list(Sentence)): A list of candidate sentences
            for the timeline summary, in decreasing preference.
        extents (list(int)): The ith entry is the extent for the ith day. If
            extent[i] = j, then no sentences in a window of j days can be
            added to the summary if the ith sentence is in the timeline.
        daily_summary_length (int): Maximum daily summary length.
        timeline_length (int): Maximum length of the timeline in days.
        start (datetime): No sentences with a date before this date can enter
            the timeline.
        end (datetime): No sentences with a date after this date can enter
            the timeline.

    Returns:
        Timeline: A timeline, greedily constructed based on `ranked_sentences`,
        respecting the constraints imposed by the other parameters.
    """
    dates_to_chosen_sentences = {}
    forbidden_dates = set()
    for i, sent in enumerate(ranked_sentences):
        date = sent.date

        if date < start or date > end:
            continue

        if date in dates_to_chosen_sentences and len(dates_to_chosen_sentences[date]) == daily_summary_length:
            continue

        if date in forbidden_dates:
            continue

        if len(dates_to_chosen_sentences) == timeline_length:
            if date in dates_to_chosen_sentences and len(dates_to_chosen_sentences[date]) < daily_summary_length:
                pass
            else:
                continue

        if extents is not None:
            forbidden_dates.add(date)
            try:
                for diff in range(1, extents[i] + 1):
                    forbidden_dates.add(date + datetime.timedelta(days=diff))
                    forbidden_dates.add(date + datetime.timedelta(days=-diff))
            except OverflowError:
                pass

        if date not in dates_to_chosen_sentences:
            dates_to_chosen_sentences[date] = []

        dates_to_chosen_sentences[date].append(sent)

    return timelines.Timeline.from_sent_objects(dates_to_chosen_sentences)
