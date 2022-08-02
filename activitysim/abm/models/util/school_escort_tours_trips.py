from activitysim.core import pipeline

import pandas as pd
import numpy as np
import warnings


def determine_chauf_outbound_flag(row, i):
    if (row['school_escort_direction'] == 'outbound'):
        outbound = True
    elif (row['school_escort_direction'] == 'inbound') & (i == 0) & (row['escort_type'] == 'pure_escort'):
        # chauf is going to pick up the first child
        outbound = True
    else:
        # chauf is inbound and has already picked up a child or taken their mandatory tour
        outbound = False
    return outbound


def create_chauf_trip_table(row):
    dropoff = True if row['school_escort_direction'] == 'outbound' else False

    row['person_id'] = row['chauf_id']
    row['destination'] = row['school_destinations'].split('_')

    participants = []
    school_escort_trip_num = []
    outbound = []
    purposes = []

    for i, child_id in enumerate(row['escortees'].split('_')):
        if dropoff:
            # have the remaining children in car
            participants.append('_'.join(row['escortees'].split('_')[i:]))
        else:
            # remaining children not yet in car
            participants.append('_'.join(row['escortees'].split('_')[:i+1]))
        school_escort_trip_num.append(i + 1)
        outbound.append(determine_chauf_outbound_flag(row, i))
        purposes.append('escort')
      
    if not dropoff:
        # adding trip home
        outbound.append(False)
        school_escort_trip_num.append(i + 2)
        purposes.append('home')
        row['destination'].append(row['home_zone_id'])
        # kids aren't in car until after they are picked up, inserting empty car for first trip
        participants = [''] + participants

    row['escort_participants'] = participants
    row['school_escort_trip_num'] = school_escort_trip_num
    row['outbound'] = outbound
    row['purpose'] = purposes
    return row


def create_chauf_escort_trips(bundles):

    chauf_trip_bundles = bundles.apply(lambda row: create_chauf_trip_table(row), axis=1)
    chauf_trip_bundles['tour_id'] = bundles['chauf_tour_id'].astype(int)

    # departure time is the first school start in the outbound school_escort_direction and the last school end in the inbound school_escort_direction
    starts = chauf_trip_bundles['school_starts'].str.split('_', expand=True).astype(float)
    ends = chauf_trip_bundles['school_ends'].str.split('_', expand=True).astype(float)
    chauf_trip_bundles['depart'] = np.where(chauf_trip_bundles['school_escort_direction'] == 'outbound', starts.min(axis=1), ends.max(axis=1))

    # create a new trip for each escortee destination
    chauf_trips = chauf_trip_bundles.explode(['destination', 'escort_participants', 'school_escort_trip_num', 'outbound', 'purpose']).reset_index()

    # numbering trips such that outbound escorting trips must come first and inbound trips must come last
    outbound_trip_num = -1 * (chauf_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=False) + 1)
    inbound_trip_num = 100 + chauf_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=True)
    chauf_trips['trip_num'] = np.where(chauf_trips.outbound == True, outbound_trip_num, inbound_trip_num)

    # --- determining trip origin    
    # origin is previous destination
    chauf_trips['origin'] = chauf_trips.groupby('tour_id')['destination'].shift()
    # outbound trips start at home
    first_outbound_trips = ((chauf_trips['outbound'] == True) & (chauf_trips['school_escort_trip_num'] == 1))
    chauf_trips.loc[first_outbound_trips, 'origin'] = chauf_trips.loc[first_outbound_trips, 'home_zone_id']
    # inbound school escort ride sharing trips start at work
    first_rs_inb = ((chauf_trips['outbound'] == False) & (chauf_trips['school_escort_trip_num'] == 1) & (chauf_trips['escort_type'] == 'ride_share'))
    chauf_trips.loc[first_rs_inb, 'origin'] = chauf_trips.loc[first_rs_inb, 'first_mand_tour_dest']

    assert all(~chauf_trips['origin'].isna()), f"Missing trip origins for {chauf_trips[chauf_trips['origin'].isna()]}"

    chauf_trips['primary_purpose'] = np.where(chauf_trips['escort_type'] == 'pure_escort', 'escort', chauf_trips['first_mand_tour_purpose'])
    assert all(~chauf_trips['primary_purpose'].isna()), f"Missing tour purpose for {chauf_trips[chauf_trips['primary_purpose'].isna()]}"

    chauf_trips['trip_id'] = chauf_trips['tour_id'].astype('int64') * 10 + chauf_trips.groupby('tour_id').cumcount()

    # trip_cols = ['trip_id', 'household_id', 'person_id', 'tour_id', 'destination', 'depart', 'escort_participants',
    #              'school_escort_trip_num', 'outbound', 'trip_num', 'primary_purpose', 'purpose', 'school_escort_direction', 'home_zone_id']
    # chauf_trips = chauf_trips[trip_cols]

    chauf_trips.loc[chauf_trips['purpose'] == 'home', 'trip_num'] = 999  # trips home are always last
    chauf_trips.sort_values(by=['household_id', 'tour_id', 'outbound', 'trip_num'], ascending=[True, True, False, True], inplace=True)

    return chauf_trips


def create_child_escorting_stops(row, escortee_num):
    escortees = row['escortees'].split('_')
    if escortee_num > (len(escortees) - 1):
        # this bundle does not have this many escortees
        return row
    dropoff = True if row['school_escort_direction'] == 'outbound' else False
    
    row['person_id'] = int(escortees[escortee_num])
    row['tour_id'] = row['school_tour_ids'].split('_')[escortee_num]
    school_dests = row['school_destinations'].split('_')

    destinations = []
    purposes = []
    participants = []
    school_escort_trip_num = []

    escortee_order = escortees[:escortee_num + 1] if dropoff else escortees[escortee_num:]

    # for i, child_id in enumerate(escortees[:escortee_num+1]):
    for i, child_id in enumerate(escortee_order):
        is_last_stop = (i == len(escortee_order) - 1)

        if dropoff:
            # dropping childen off
            # children in car are the child and the children after
            participants.append('_'.join(escortees[i:]))
            dest = school_dests[i]
            purpose = 'school' if row['person_id'] == int(child_id) else 'escort'

        else:
            # picking children up
            # children in car are the child and those already picked up
            participants.append('_'.join(escortees[:escortee_num + i +1]))
            # going home if last stop, otherwise to next school destination
            dest = row['home_zone_id'] if is_last_stop else school_dests[i+1]
            purpose = 'home' if is_last_stop else 'escort'
            
        
        # filling arrays
        destinations.append(dest)
        school_escort_trip_num.append(i + 1)
        purposes.append(purpose)

    row['escort_participants'] = participants
    row['school_escort_trip_num'] = school_escort_trip_num
    row['purpose'] = purposes
    row['destination'] = destinations
    return row


def create_escortee_trips(bundles):

    escortee_trips = []
    for escortee_num in range(0, bundles.num_escortees.max() + 1):
        escortee_bundles = bundles.apply(lambda row: create_child_escorting_stops(row, escortee_num), axis=1)
        escortee_trips.append(escortee_bundles)

    escortee_trips = pd.concat(escortee_trips)
    escortee_trips = escortee_trips[~escortee_trips.person_id.isna()]

    # departure time is the first school start in the outbound direction and the last school end in the inbound direction
    starts = escortee_trips['school_starts'].str.split('_', expand=True).astype(float)
    ends = escortee_trips['school_ends'].str.split('_', expand=True).astype(float)
    escortee_trips['outbound'] = np.where(escortee_trips['school_escort_direction'] == 'outbound', True, False)
    escortee_trips['depart'] = np.where(escortee_trips['school_escort_direction'] == 'outbound', starts.min(axis=1), ends.max(axis=1)).astype(int)
    escortee_trips['primary_purpose'] = 'school'

    # create a new trip for each escortee destination
    escortee_trips = escortee_trips.explode(['destination', 'escort_participants', 'school_escort_trip_num', 'purpose']).reset_index()

    # numbering trips such that outbound escorting trips must come first and inbound trips must come last
    outbound_trip_num = -1 * (escortee_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=False) + 1)
    inbound_trip_num = 100 + escortee_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=True)
    escortee_trips['trip_num'] = np.where(escortee_trips.outbound == True, outbound_trip_num, inbound_trip_num)
    escortee_trips['trip_count'] = escortee_trips['trip_num'] + escortee_trips.groupby(['tour_id', 'outbound']).trip_num.transform('count')

    # FIXME placeholders
    escortee_trips['trip_id'] = escortee_trips['tour_id'].astype('int64') * 10 + escortee_trips.groupby('tour_id')['trip_num'].cumcount()

    # trip_cols = ['trip_id', 'household_id', 'person_id', 'tour_id', 'destination', 'depart', 'escort_participants',
    #             'school_escort_trip_num', 'outbound', 'primary_purpose', 'purpose', 'school_escort_direction', 'trip_num', 'home_zone_id']
    # escortee_trips = escortee_trips[trip_cols]

    id_cols = ['household_id', 'person_id', 'tour_id']
    escortee_trips[id_cols] = escortee_trips[id_cols].astype(int)
    # for col in escortee_trips.columns:
    #     if col in ['trip_id', 'household_id', 'person_id', 'tour_id']:
    #         escortee_trips[col] = escortee_trips[col].astype(int)

    escortee_trips.loc[escortee_trips['purpose'] == 'home', 'trip_num'] = 999  # trips home are always last
    escortee_trips.sort_values(by=['household_id', 'tour_id', 'outbound', 'trip_num'], ascending=[True, True, False, True], inplace=True)
    escortee_trips['origin'] = escortee_trips.groupby('tour_id')['destination'].shift()
    # first trips on tour start from home (except for atwork subtours, but school escorting doesn't happen on those tours)
    escortee_trips['origin'] = np.where(escortee_trips['origin'].isna(), escortee_trips['home_zone_id'], escortee_trips['origin'])
    
    return escortee_trips


def create_school_escort_trips(escort_bundles):
    chauf_trips = create_chauf_escort_trips(escort_bundles)
    assert all(chauf_trips.trip_id > 0), f"Negative trip_id's {chauf_trips[chauf_trips.trip_id < 0]}"
    escortee_trips = create_escortee_trips(escort_bundles)
    assert all(escortee_trips.trip_id > 0), f"Negative trip_id's {escortee_trips[escortee_trips.trip_id < 0]}"
    school_escort_trips = pd.concat([chauf_trips, escortee_trips], axis=0)

    school_escort_trips['failed'] = False  # for better merge with trips created in stop frequency
    school_escort_trips.set_index('trip_id', inplace=True)

    assert school_escort_trips.index.is_unique, f"Non-unique trip_id's set as index {school_escort_trips[school_escort_trips.index.duplicated(keep=False)]}"
    assert all(school_escort_trips.index > 0), f"Negative trip_id's {school_escort_trips[school_escort_trips.index < 0]}"

    return school_escort_trips
    

def add_pure_escort_tours(tours, school_escort_tours):
    missing_cols = [col for col in tours.columns if col not in school_escort_tours.columns]
    assert len(missing_cols) == 0, f'missing columns {missing_cols} in school_escort_tours'

    tours_to_add = school_escort_tours[~school_escort_tours.index.isin(tours.index)]
    tours = pd.concat([tours, tours_to_add[tours.columns]])
    return tours


def add_school_escort_trips_to_pipeline():
    school_escort_trips = pipeline.get_table('school_escort_trips')
    tours = pipeline.get_table('tours')
    trips = pipeline.get_table('trips')
    
    # want to remove stops if school escorting takes place on that half tour so we can replace them with the actual stops
    out_se_tours = tours[tours['school_esc_outbound'].isin(['pure_escort', 'ride_share'])]
    inb_se_tours = tours[tours['school_esc_inbound'].isin(['pure_escort', 'ride_share'])]
    # removing outbound stops
    trips = trips[~(trips.tour_id.isin(out_se_tours.index) & (trips['outbound'] == True))]
    # removing inbound stops
    trips = trips[~(trips.tour_id.isin(inb_se_tours.index) & (trips['outbound'] == False))]

    # don't want to double count the non-escort half-tour of chauffeurs doing pure escort
    inb_chauf_pe_tours = tours[(tours['school_esc_inbound'] == 'pure_escort') & (tours.primary_purpose == 'escort')]
    out_chauf_pe_tours = tours[(tours['school_esc_outbound'] == 'pure_escort') & (tours.primary_purpose == 'escort')]
    school_escort_trips = school_escort_trips[~(school_escort_trips.tour_id.isin(inb_chauf_pe_tours.index) & (school_escort_trips['outbound'] == True))]
    school_escort_trips = school_escort_trips[~(school_escort_trips.tour_id.isin(out_chauf_pe_tours.index) & (school_escort_trips['outbound'] == False))]

    trips = pd.concat([trips, school_escort_trips[list(trips.columns) + ['escort_participants', 'school_escort_direction']]])
    # sorting by escorting order as determining when creating the school escort trips
    trips.sort_values(by=['household_id', 'tour_id', 'outbound', 'trip_num'], ascending=[True, True, False, True], inplace=True)
    grouped = trips.groupby(['tour_id', 'outbound'])
    trips['trip_num'] = trips.groupby(['tour_id', 'outbound']).cumcount() + 1
    trips['trip_count'] = trips['trip_num'] + grouped.cumcount(ascending=False)

    # ensuring data types
    trips['outbound'] = trips['outbound'].astype(bool)
    print(trips[trips['origin'].isna()]) 
    trips['origin'] = trips['origin'].astype(int)
    trips['destination'] = trips['destination'].astype(int)

    assert trips.index.is_unique, f"Non-unique trip_id's set as index {trips[trips.index.duplicated(keep=False)]}"

    # replace trip table and pipeline and register with the random number generator
    pipeline.replace_table("trips", trips)
    pipeline.get_rn_generator().drop_channel('trips')
    pipeline.get_rn_generator().add_channel('trips', trips)

    # FIXME need to update stop frequency in tours table
    
    return trips



def create_pure_school_escort_tours(bundles):
    # creating home to school tour for chauffers making pure escort tours
    # ride share tours are already created since they go off the mandatory tour

    # FIXME: can I just move all of this logic to a csv and annotate??
    # bundles = pipeline.get_table('escort_bundles')
    persons = pipeline.get_table('persons')
    pe_tours = bundles[bundles['escort_type'] == 'pure_escort']

    pe_tours['origin'] = pe_tours['home_zone_id']
    # desination is the last dropoff / pickup location
    pe_tours['destination'] = pe_tours['school_destinations'].str.split('_').str[-1].astype(int)
    # start is the first start time
    pe_tours['start'] = pe_tours['school_starts'].str.split('_').str[0].astype(int)

    school_time_cols = ['time_home_to_school' + str(i) for i in range(1,4)]
    # FIXME hard coded mins per time bin, is rounding down appropriate?
    pe_tours['end'] = pe_tours['start'] + (pe_tours[school_time_cols].sum(axis=1) / 30).astype(int)

    pe_tours['person_id'] = pe_tours['chauf_id']
    # FIXME should probably put this when creating the bundles table
    assert all(pe_tours['person_id'].isin(persons.index)), \
        f"Chauffer ID(s) not present in persons table {pe_tours.loc[~pe_tours['person_id'].isin(persons.index), 'person_id']}"
    # pe_tours = pe_tours[pe_tours['person_id'].isin(persons.index)]

    pe_tours['tour_category'] = 'non_mandatory'
    pe_tours['number_of_participants'] = 1
    pe_tours['tour_type'] = 'escort'
    # FIXME join tdd from tdd_alts
    pe_tours['tdd'] = pd.NA
    pe_tours['duration'] = pe_tours['end'] - pe_tours['start']
    pe_tours['school_esc_outbound'] = np.where(pe_tours['school_escort_direction'] == 'outbound', 'pure_escort', pd.NA)
    pe_tours['school_esc_inbound'] = np.where(pe_tours['school_escort_direction'] == 'inbound', 'pure_escort', pd.NA)

    pe_tours['tour_id'] = pe_tours['chauf_tour_id'].astype(int)
    pe_tours.set_index('tour_id', inplace=True)

    grouped = pe_tours.groupby(['person_id', 'tour_type'])
    pe_tours['tour_type_num'] = grouped.cumcount() + 1
    pe_tours['tour_type_count'] = pe_tours['tour_type_num'] + grouped.cumcount(ascending=False)

    grouped = pe_tours.groupby('person_id')
    pe_tours['tour_num'] = grouped.cumcount() + 1
    pe_tours['tour_count'] = pe_tours['tour_num'] + grouped.cumcount(ascending=False)

    return pe_tours
