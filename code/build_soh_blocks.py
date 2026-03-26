import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from config import ARTIFACTS_DIR

filepath = os.path.join(ARTIFACTS_DIR, "cycles.csv")
outpath  = os.path.join(ARTIFACTS_DIR, "capacity_soh_blocks.csv")

REF_CAPACITY_AH      = 436.0  # full-pack nominal (2 strings × 218 Ah)
CHARGE_COULOMBIC_EFF = 0.97   # LiNMC ~97% coulombic efficiency
PARALLEL_STRINGS     = 2      # charger current sensor measures both strings combined


rows = []
with open(filepath, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

vehicles = defaultdict(list)
for r in rows:
    vehicles[r['registration_number']].append(r)

results = []


def emit_discharge(reg, st):
    soc_s      = st['cur_soc_start']
    soc_e      = st['cur_soc_end']
    gross      = st['cur_discharge_gross']
    energy_kwh = st['cur_energy_kwh']
    dod        = (float(soc_s) - float(soc_e)) / 100.0 if soc_s and soc_e else None
    if not (dod and dod >= 0.30 and gross > 0):
        return
    norm_cap          = gross / dod
    raw_soh           = round(norm_cap / REF_CAPACITY_AH * 100, 2)
    capped            = raw_soh > 100.0
    energy_normalized = round(energy_kwh / dod, 4) if energy_kwh else None
    st['block_num'] += 1
    results.append({
        'registration_number':          reg,
        'block_num':                    st['block_num'],
        'block_type':                   'discharge',
        'soc_start':                    soc_s,
        'soc_end':                      soc_e,
        'dod_or_doc_pct':               round(dod * 100, 1),
        # discharge capacity
        'capacity_ah_discharge':        round(gross, 4),
        # charging capacity (blank for discharge)
        'capacity_ah_charge_raw':       '',
        'capacity_ah_charge_corrected': '',
        # energy
        'energy_kwh':                   round(energy_kwh, 4) if energy_kwh else None,
        'energy_kwh_normalized':        energy_normalized,
        # SOH
        'ref_capacity_ah':              REF_CAPACITY_AH,
        'normalized_capacity_ah':       round(norm_cap, 4),
        'capacity_soh_pct':             100.0 if capped else raw_soh,
        'soh_capped':                   capped,
        'actual_soh_pct':               raw_soh if capped else '',
        'n_sessions':                   st['cur_n'],
        'start_time_ist':               st['cur_start_ist'],
        'end_time_ist':                 st['cur_end_ist'],
    })


def emit_charging(reg, st, s, cap_chg_raw, voltage_mean):
    soc_s = s['soc_start']
    soc_e = s['soc_end']
    doc   = (float(soc_e) - float(soc_s)) / 100.0 if soc_s and soc_e else None

    # Correct for 2-string current measurement
    cap_chg_corrected = cap_chg_raw / PARALLEL_STRINGS

    # Energy using corrected Ah
    energy_kwh = round(cap_chg_corrected * voltage_mean / 1000, 4) if voltage_mean else None

    if doc and doc > 0.01 and cap_chg_corrected > 0:
        norm_chg          = (cap_chg_corrected * CHARGE_COULOMBIC_EFF) / doc
        raw_soh           = round(norm_chg / REF_CAPACITY_AH * 100, 2)
        capped            = raw_soh > 100.0
        energy_normalized = round(energy_kwh / doc, 4) if energy_kwh else None
    else:
        norm_chg          = None
        raw_soh           = None
        capped            = False
        energy_normalized = None

    st['block_num'] += 1
    results.append({
        'registration_number':          reg,
        'block_num':                    st['block_num'],
        'block_type':                   'charging',
        'soc_start':                    soc_s,
        'soc_end':                      soc_e,
        'dod_or_doc_pct':               round(doc * 100, 1) if doc else None,
        # discharge capacity (blank for charging)
        'capacity_ah_discharge':        '',
        # charging capacity
        'capacity_ah_charge_raw':       round(cap_chg_raw, 4),
        'capacity_ah_charge_corrected': round(cap_chg_corrected, 4),
        # energy
        'energy_kwh':                   energy_kwh,
        'energy_kwh_normalized':        energy_normalized,
        # SOH
        'ref_capacity_ah':              REF_CAPACITY_AH,
        'normalized_capacity_ah':       round(norm_chg, 4) if norm_chg else None,
        'capacity_soh_pct':             (100.0 if capped else raw_soh) if raw_soh else None,
        'soh_capped':                   capped if raw_soh is not None else '',
        'actual_soh_pct':               raw_soh if capped else '',
        'n_sessions':                   1,
        'start_time_ist':               s['start_time_ist'],
        'end_time_ist':                 s['end_time_ist'],
    })


for reg, sessions in vehicles.items():
    sessions.sort(key=lambda x: int(x['start_time']))

    state = {
        'block_num':           0,
        'cur_type':            None,
        'cur_discharge_gross': 0.0,
        'cur_energy_kwh':      0.0,
        'cur_start_ist':       None,
        'cur_end_ist':         None,
        'cur_soc_start':       None,
        'cur_soc_end':         None,
        'cur_n':               0,
    }

    for s in sessions:
        stype         = s['session_type']
        cap_dis       = float(s['capacity_ah_discharge'])    if s['capacity_ah_discharge']    else 0.0
        cap_chg_raw   = float(s['capacity_ah_charge_total']) if s['capacity_ah_charge_total'] else 0.0
        energy_kwh    = float(s['energy_kwh'])               if s['energy_kwh']               else 0.0
        voltage_mean  = float(s['voltage_mean'])             if s['voltage_mean']             else None

        if stype == 'charging':
            if state['cur_type'] == 'discharge':
                emit_discharge(reg, state)
            state.update({'cur_type': 'charging', 'cur_discharge_gross': 0.0,
                          'cur_energy_kwh': 0.0, 'cur_n': 0,
                          'cur_soc_start': None, 'cur_soc_end': None})
            emit_charging(reg, state, s, cap_chg_raw, voltage_mean)

        elif stype == 'discharge':
            if state['cur_type'] != 'discharge':
                state.update({
                    'cur_type':            'discharge',
                    'cur_discharge_gross': cap_dis,
                    'cur_energy_kwh':      energy_kwh,
                    'cur_start_ist':       s['start_time_ist'],
                    'cur_end_ist':         s['end_time_ist'],
                    'cur_soc_start':       s['soc_start'],
                    'cur_soc_end':         s['soc_end'],
                    'cur_n':               1,
                })
            else:
                state['cur_discharge_gross'] += cap_dis
                state['cur_energy_kwh']      += energy_kwh
                state['cur_end_ist']          = s['end_time_ist']
                state['cur_soc_end']          = s['soc_end']
                state['cur_n']               += 1

        elif stype == 'idle':
            if state['cur_type'] == 'discharge':
                state['cur_discharge_gross'] += cap_dis
                state['cur_energy_kwh']      += energy_kwh
                state['cur_end_ist']          = s['end_time_ist']
                state['cur_soc_end']          = s['soc_end']
                state['cur_n']               += 1

    if state['cur_type'] == 'discharge':
        emit_discharge(reg, state)


fieldnames = [
    'registration_number', 'block_num', 'block_type',
    'soc_start', 'soc_end', 'dod_or_doc_pct',
    'capacity_ah_discharge',
    'capacity_ah_charge_raw', 'capacity_ah_charge_corrected',
    'energy_kwh', 'energy_kwh_normalized',
    'ref_capacity_ah', 'normalized_capacity_ah',
    'capacity_soh_pct', 'soh_capped', 'actual_soh_pct',
    'n_sessions', 'start_time_ist', 'end_time_ist',
]

with open(outpath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

dis = [r for r in results if r['block_type'] == 'discharge']
chg = [r for r in results if r['block_type'] == 'charging']
dis_capped = sum(1 for r in dis if r['soh_capped'] is True)
chg_capped = sum(1 for r in chg if r['soh_capped'] is True)
chg_valid  = [r for r in chg if r['capacity_soh_pct'] is not None]

print(f'Total rows       : {len(results)}')
print(f'Discharge blocks : {len(dis)}  (capped: {dis_capped})')
print(f'Charging sessions: {len(chg)}  (capped: {chg_capped}, no-data: {len(chg)-len(chg_valid)})')
print()

print('Sample - MH18BZ2647 first 8 rows:')
sample = [r for r in results if r['registration_number'] == 'MH18BZ2647'][:8]
print(f"{'Blk':<5} {'Type':<10} {'SOC s->e':<14} {'DoD/C%':<7} {'Dis_Ah':<9} {'Chg_raw':<9} {'Chg_cor':<9} {'E_kwh':<8} {'E_norm':<9} {'Norm_Ah':<9} {'SOH%':<8} {'Capped'}")
print('-' * 115)
for r in sample:
    soc = f"{r['soc_start']}%->{r['soc_end']}%"
    print(f"{r['block_num']:<5} {r['block_type']:<10} {soc:<14} "
          f"{str(r['dod_or_doc_pct'])+'%':<7} "
          f"{str(r['capacity_ah_discharge']):<9} "
          f"{str(r['capacity_ah_charge_raw']):<9} "
          f"{str(r['capacity_ah_charge_corrected']):<9} "
          f"{str(r['energy_kwh']):<8} "
          f"{str(r['energy_kwh_normalized']):<9} "
          f"{str(r['normalized_capacity_ah']):<9} "
          f"{str(r['capacity_soh_pct']):<8} {r['soh_capped']}")
print()
print(f'Saved: {outpath}')
