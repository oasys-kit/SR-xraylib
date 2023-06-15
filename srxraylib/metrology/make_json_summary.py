# this scrips make a summary file pf the dabam contents to accelerate the search
if __name__ == '__main__':
    from srxraylib.metrology.dabam import make_json_summary
    make_json_summary(force_from_scratch=True, server="/users/srio/OASYS1.2/DabamFiles")