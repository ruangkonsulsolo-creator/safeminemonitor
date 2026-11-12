# Ground truth for event-level validation
One CSV per test video. Watch the ORIGINAL video (not the annotated one) and write
one row per continuous unsafe behaviour of one worker:

    category,start_s,end_s,note
    NO_HELMET,3.0,9.5,worker left
    RESTRICTED_ZONE,12.0,15.0,

Categories: NO_HELMET, NO_VEST, RESTRICTED_ZONE, MACHINE_PROXIMITY.
Upload it on the Validation page (select the processed job). Matching rule:
same category and temporal IoU >= 0.3 (one-to-one). Ideally two people label
independently and disagreements are resolved by discussion.
