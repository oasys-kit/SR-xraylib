"""
This program makes a summary file pf the dabam contents to accelerate the search.

Example
-------
>>> python -m srxraylib.metrology.make_json_summary
"""
if __name__ == '__main__':
    """
    Main program to make a summary of dabam profiles.

    Example
    -------
    >>> python -m srxraylib.metrology.make_json_summary  # writes the file dabam-summary.json
    
    """
    from srxraylib.metrology.dabam import make_json_summary
    make_json_summary(force_from_scratch=True) # , server="/users/srio/OASYS1.2/DabamFiles")