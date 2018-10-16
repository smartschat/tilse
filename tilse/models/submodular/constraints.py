"""
Contains functions describing constraints for submodular optimization.
"""


def is_valid_total_length(index,
                          date_to_sent_mapping,
                          all_sent_dates,
                          timeline_properties):
    """
    Checks whether adding the sentence in focus would not violate
    the constraint of limiting the total number of sentences in a
    timeline.

    Corresponds to the AsMDS constrained described in Martschat and Markert
    (2018).
    
    Params:
        index (int): Index of the sentence in focus.
        date_to_sent_mapping (dict(datetime.datetime, list(tilse.data.sentences.Sentence)): 
            Mapping of dates in a timeline to the sentences in the summary for this date
            (describes the partial timeline constructed so far).
        all_sent_dates (list(datetime.dateime)): Dates for all sentences. In particular,
            `all_sent_dates[index]` is the date of the sentence in focus.
        timeline_properties (tilse.models.timeline_properties.TimelineProperties):
            Properties of the timeline to predict.
            
    Returns:
        False if (i) date of the sentence in focus is before start or after end date of the 
        timeline as defined in `timeline_properties` or (ii) adding the sentence in focus
        would lead to a timeline with more sentences than `timeline_properties.num_sentences`;
        True otherwise.
    """
    selected_date = all_sent_dates[index]

    if selected_date < timeline_properties.start \
            or selected_date > timeline_properties.end:
        return False

    return sum([len(sents) for sents in date_to_sent_mapping.values()]) \
           < timeline_properties.num_sentences


def is_valid_individual_constraints(index,
                                    date_to_sent_mapping,
                                    all_sent_dates,
                                    timeline_properties):
    """
    Checks whether adding the sentence in focus would not violate
    the constraint of limiting the number of days and the length
    of daily summaries in the timeline.

    Corresponds to the TLSConstraints constraints described in Martschat and
    Markert (2018).
    
    Params:
        index (int): Index of the sentence in focus.
        date_to_sent_mapping (dict(datetime.datetime, list(tilse.data.sentences.Sentence)): 
            Mapping of dates in a timeline to the sentences in the summary for this date
            (describes the partial timeline constructed so far).
        all_sent_dates (list(datetime.dateime)): Dates for all sentences. In particular,
            `all_sent_dates[index]` is the date of the sentence in focus.
        timeline_properties (tilse.models.timeline_properties.TimelineProperties):
            Properties of the timeline to predict.
            
    Returns:
        False if (i) date of the sentence in focus is before start or after end date of the 
        timeline as defined in `timeline_properties` or (ii) adding the sentence in focus
        would lead to a timeline with more dates than `timeline_properties.num_dates` or 
        (iii) adding the sentence in focus would lead to a timeline that has a daily summary 
        longer than `timeline_properties.daily_summary_length` sentences; True otherwise.
    """                                    
                                    
    summary_length = timeline_properties.daily_summary_length
    desired_timeline_length = timeline_properties.num_dates

    selected_date = all_sent_dates[index]

    if selected_date < timeline_properties.start \
            or selected_date > timeline_properties.end:
        return False
    elif len(date_to_sent_mapping) == desired_timeline_length \
            and selected_date not in date_to_sent_mapping:
        return False
    elif selected_date in date_to_sent_mapping \
            and len(date_to_sent_mapping[selected_date]) == summary_length:
        return False
    else:
        return True
