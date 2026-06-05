import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import os
import datetime

# --- CONFIG & CSS ---
st.set_page_config(page_title="Coach Pro", layout="wide", initial_sidebar_state="collapsed")

# Hide default Streamlit clutter to make it look like a native app
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container { padding-top: 1rem; padding-bottom: 1rem; }
            /* Make buttons expand to fill columns */
            div.stButton > button:first-child { width: 100%; height: 60px; }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- CONSTANTS ---
ROSTER_FILE = "team_roster.json"
HISTORY_FILE = "games_history.json"
POSITIONS = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
SPOTS = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "EH1", "EH2", "EH3"]

DEFAULT_ROSTER = {
    "Player_1": {"Active": True, "P": 8, "C": 0, "1B": 4, "2B": 9, "3B": 7, "SS": 8, "LF": 5, "CF": 5, "RF": 5},
    "Player_2": {"Active": True, "P": 7, "C": 0, "1B": 5, "2B": 6, "3B": 8, "SS": 7, "LF": 6, "CF": 4, "RF": 5},
    "Player_3": {"Active": True, "P": 0, "C": 9, "1B": 2, "2B": 0, "3B": 0, "SS": 0, "LF": 0, "CF": 0, "RF": 0},
    "Player_4": {"Active": True, "P": 0, "C": 0, "1B": 8, "2B": 4, "3B": 5, "SS": 4, "LF": 3, "CF": 2, "RF": 3},
    "Player_5": {"Active": True, "P": 4, "C": 0, "1B": 4, "2B": 3, "3B": 4, "SS": 5, "LF": 8, "CF": 8, "RF": 7},
    "Player_6": {"Active": True, "P": 5, "C": 0, "1B": 3, "2B": 5, "3B": 6, "SS": 6, "LF": 6, "CF": 9, "RF": 7},
    "Player_7": {"Active": True, "P": 2, "C": 0, "1B": 5, "2B": 4, "3B": 4, "SS": 5, "LF": 7, "CF": 7, "RF": 9},
    "Player_8": {"Active": True, "P": 0, "C": 6, "1B": 7, "2B": 4, "3B": 6, "SS": 5, "LF": 5, "CF": 4, "RF": 5},
    "Player_9": {"Active": True, "P": 3, "C": 0, "1B": 4, "2B": 8, "3B": 6, "SS": 9, "LF": 6, "CF": 7, "RF": 6},
    "Player_10": {"Active": True, "P": 0, "C": 0, "1B": 4, "2B": 5, "3B": 4, "SS": 5, "LF": 6, "CF": 5, "RF": 6},
    "Player_11": {"Active": True, "P": 0, "C": 0, "1B": 6, "2B": 3, "3B": 5, "SS": 3, "LF": 7, "CF": 6, "RF": 7},
    "Player_12": {"Active": False, "P": 0, "C": 0, "1B": 5, "2B": 4, "3B": 4, "SS": 4, "LF": 5, "CF": 5, "RF": 5},
}

# --- STATE MANAGEMENT ---
def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except: pass
    return default

if 'roster' not in st.session_state: st.session_state.roster = load_json(ROSTER_FILE, DEFAULT_ROSTER)
if 'history' not in st.session_state: st.session_state.history = load_json(HISTORY_FILE, [])
if 'on_field' not in st.session_state: st.session_state.on_field = {}
if 'bench_counts' not in st.session_state: st.session_state.bench_counts = {p: 0 for p in st.session_state.roster.keys()}
if 'current_inning' not in st.session_state: st.session_state.current_inning = 1
if 'game_log' not in st.session_state: st.session_state.game_log = []
if 'selected_sub_out' not in st.session_state: st.session_state.selected_sub_out = None # Tracks the Tap-to-Sub state

def get_active_players(): return [p for p, data in st.session_state.roster.items() if data.get('Active', True)]
def get_current_bench():
    fielded = [data['player'] for data in st.session_state.on_field.values()] if st.session_state.on_field else []
    return [p for p in get_active_players() if p not in fielded]
def log_event(text): st.session_state.game_log.append(f"[Inning {st.session_state.current_inning}] {text}")

# --- ALGORITHM LOGIC ---
def calculate_optimal_lineup(active_pitcher, simulate_future=False, max_limit=2):
    cost_matrix = []
    active = get_active_players()
    orig_p_rating = st.session_state.roster[active_pitcher].get("P", 0)
    st.session_state.roster[active_pitcher]["P"] = 100 
    
    for player in active:
        row = []
        eff_sat = st.session_state.bench_counts.get(player, 0)
        if simulate_future and player in get_current_bench(): eff_sat += 1
        forced_play = eff_sat >= max_limit
        
        for pos in POSITIONS:
            skill = st.session_state.roster[player].get(pos, 0)
            if skill == 0: cost = 99999
            else:
                cost = 100 - skill - (eff_sat * 15) 
                if forced_play: cost -= 5000
            row.append(cost)
        cost_matrix.append(row)
        
    row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
    st.session_state.roster[active_pitcher]["P"] = orig_p_rating 
    
    results, total_skill = {}, 0
    for i in range(len(col_ind)):
        p = active[row_ind[i]]
        pos = POSITIONS[col_ind[i]]
        skill = st.session_state.roster[p].get(pos, 0)
        total_skill += skill
        results[pos] = {"player": p, "skill": skill}
    return results, total_skill

def generate_instructions(old_map, new_map):
    if not old_map: return "Starting lineup locked."
    instructions = []
    new_p, old_p = new_map.get("P", {}).get("player"), old_map.get("P", {}).get("player")
    if new_p and new_p != old_p: instructions.append(f"⚾ {new_p} enters to pitch.")
    for pos, data in new_map.items():
        if pos == "P": continue
        p = data["player"]
        old_pos = next((op for op, d in old_map.items() if d["player"] == p), None)
        if old_pos and old_pos != pos: instructions.append(f"🔄 {p} moves: {old_pos} -> {pos}.")
        elif not old_pos: instructions.append(f"⬆️ {p} enters to play {pos}.")
    for pos, data in old_map.items():
        if data["player"] not in [d["player"] for d in new_map.values()]: instructions.append(f"⬇️ {data['player']} moves to Bench.")
    return "\n".join(instructions) if instructions else "No defensive changes."

# --- INTERACTIVE TAP-TO-SUB CONTROLLERS ---
def handle_field_click(pos):
    if st.session_state.selected_sub_out == pos:
        st.session_state.selected_sub_out = None # Deselect if tapped twice
    else:
        st.session_state.selected_sub_out = pos

def handle_bench_click(player_in):
    if st.session_state.selected_sub_out:
        pos_out = st.session_state.selected_sub_out
        player_out = st.session_state.on_field[pos_out]["player"]
        
        # Execute the swap
        skill_in = st.session_state.roster[player_in].get(pos_out, 0)
        st.session_state.on_field[pos_out] = {"player": player_in, "skill": skill_in}
        
        log_event(f"Manual Sub: {player_in} in for {player_out} at {pos_out}")
        st.session_state.selected_sub_out = None # Reset state machine
        st.rerun()

# --- UI TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Roster", "🏁 Start Lineup", "⚾ Game Day", "📚 History"])

# ==========================================
# TAB 1: ROSTER MANAGEMENT
# ==========================================
with tab1:
    st.header("Manage Team Roster")
    df = pd.DataFrame.from_dict(st.session_state.roster, orient='index')
    df = df[['Active'] + POSITIONS]
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Roster", type="primary"):
        st.session_state.roster = edited_df.to_dict(orient='index')
        with open(ROSTER_FILE, 'w') as f: json.dump(st.session_state.roster, f)
        for p in st.session_state.roster.keys():
            if p not in st.session_state.bench_counts: st.session_state.bench_counts[p] = 0
        st.success("Roster Saved!")

# ==========================================
# TAB 2: STARTING LINEUP
# ==========================================
with tab2:
    st.header("Set Starting Lineup")
    active_players = [""] + get_active_players()
    cols = st.columns(3)
    start_selections = {}
    for i, spot in enumerate(SPOTS):
        with cols[i % 3]:
            start_selections[spot] = st.selectbox(f"{spot}", options=active_players, key=f"start_{spot}")
            
    if st.button("✅ Lock Lineup & Play Ball!", type="primary"):
        manual_lineup = {}
        error = False
        for pos in POSITIONS:
            p = start_selections[pos]
            if not p: st.error(f"Please assign a player to {pos}"); error = True; break
            manual_lineup[pos] = {"player": p, "skill": st.session_state.roster[p].get(pos, 0)}
        if not error:
            st.session_state.on_field = manual_lineup
            log_event("Starting Lineup Locked.")
            st.success("Lineup Locked! Go to Game Day tab.")

# ==========================================
# TAB 3: GAME DAY (MOBILE OPTIMIZED)
# ==========================================
with tab3:
    col1, col2 = st.columns([2, 1])
    with col1: st.header(f"Inning: {st.session_state.current_inning}")
    with col2: max_bench = st.number_input("Max Innings on Bench:", min_value=1, value=2)

    # ACTION BAR
    c_sub, c_adv, c_end = st.columns(3)
    with c_sub:
        fielded_pitcher = st.session_state.on_field.get("P", {}).get("player", "") if st.session_state.on_field else ""
        new_pitcher = st.selectbox("Sub Pitcher:", options=get_active_players(), index=get_active_players().index(fielded_pitcher) if fielded_pitcher in get_active_players() else 0)
        if st.button("🔄 Exec Pitch Change"):
            if new_pitcher and st.session_state.on_field:
                new_lineup, _ = calculate_optimal_lineup(new_pitcher, max_limit=max_bench)
                log_event(f"Pitching Change:\n{generate_instructions(st.session_state.on_field, new_lineup)}")
                st.session_state.on_field = new_lineup; st.rerun()

    with c_adv:
        auto_rotate = st.checkbox("Auto-Apply Rotations", value=True)
        if st.button("⏩ Adv. Inning", type="primary"):
            if st.session_state.on_field:
                for p in get_current_bench(): st.session_state.bench_counts[p] += 1
                st.session_state.current_inning += 1
                optimal, _ = calculate_optimal_lineup(new_pitcher, max_limit=max_bench)
                if auto_rotate:
                    log_event(f"Auto-Rotated:\n{generate_instructions(st.session_state.on_field, optimal)}")
                    st.session_state.on_field = optimal
                else: log_event("Inning advanced manually.")
                st.rerun()

    with c_end:
        if st.button("🛑 End Game"):
            timestamp = datetime.datetime.now().strftime("%b %d, %Y - %I:%M %p")
            summary = [f"Game: {timestamp}", f"Total Innings: {st.session_state.current_inning}", "\nFINAL BENCH COUNTS:"]
            for p in get_active_players(): summary.append(f"• {p}: {st.session_state.bench_counts.get(p, 0)}")
            summary.append("\nPLAY-BY-PLAY:"); summary.extend(st.session_state.game_log)
            st.session_state.history.insert(0, {"name": timestamp, "summary": "\n".join(summary)})
            with open(HISTORY_FILE, 'w') as f: json.dump(st.session_state.history, f)
            st.session_state.on_field = {}; st.session_state.current_inning = 1; st.session_state.game_log = []; st.session_state.selected_sub_out = None
            for p in st.session_state.bench_counts: st.session_state.bench_counts[p] = 0
            st.success("Game Saved!"); st.rerun()

    st.markdown("---")

    # --- VISUAL BASEBALL DIAMOND ---
    if st.session_state.on_field:
        st.markdown("<h4 style='text-align: center;'>The Diamond</h4>", unsafe_allow_html=True)
        
        def render_pos(pos):
            player = st.session_state.on_field.get(pos, {}).get("player", "--")
            # If selected, make it a primary (red) button
            btn_type = "primary" if st.session_state.selected_sub_out == pos else "secondary"
            btn_label = f"SUB OUT {pos}" if st.session_state.selected_sub_out == pos else f"{pos}\n{player}"
            st.button(btn_label, key=f"fld_{pos}", on_click=handle_field_click, args=(pos,), type=btn_type)

        # Row 1: Center Field
        c1, c2, c3, c4, c5 = st.columns(5)
        with c3: render_pos("CF")
        
        # Row 2: LF, SS, 2B, RF
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: render_pos("LF")
        with c2: render_pos("SS")
        with c4: render_pos("2B")
        with c5: render_pos("RF")
        
        # Row 3: 3B, P, 1B
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: render_pos("3B")
        with c3: render_pos("P")
        with c5: render_pos("1B")
        
        # Row 4: Catcher
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2: render_pos("C")
            
        st.markdown("---")
        
        # --- THE BENCH (Interactive) ---
        bench_players = get_current_bench()
        
        if st.session_state.selected_sub_out:
            st.warning(f"Select a player from the bench below to sub into {st.session_state.selected_sub_out}:")
        else:
            st.info("Current Bench (Tap a player on the field above to initiate a sub)")
            
        b_cols = st.columns(3)
        for i, p in enumerate(bench_players):
            sat = st.session_state.bench_counts.get(p, 0)
            btn_label = f"SUB IN: {p}" if st.session_state.selected_sub_out else f"{p} (Sat: {sat})"
            btn_type = "primary" if st.session_state.selected_sub_out else "secondary"
            
            with b_cols[i % 3]:
                st.button(btn_label, key=f"bnch_{p}", on_click=handle_bench_click, args=(p,), type=btn_type)
        
        st.markdown("---")
        st.write("📝 **Recent Moves:**")
        if st.session_state.game_log:
            st.text(st.session_state.game_log[-1])
            
        st.write("🔮 **Predicted Next Inning Moves:**")
        future, _ = calculate_optimal_lineup(new_pitcher, simulate_future=True, max_limit=max_bench)
        st.code(generate_instructions(st.session_state.on_field, future))
        
    else:
        st.warning("Please lock your starting lineup on the previous tab.")

# ==========================================
# TAB 4: HISTORY
# ==========================================
with tab4:
    st.header("Games History")
    if not st.session_state.history:
        st.info("No saved games yet.")
    else:
        game_names = [g["name"] for g in st.session_state.history]
        selected_game = st.selectbox("Select Game:", game_names)
        
        if selected_game:
            game_data = next((g for g in st.session_state.history if g["name"] == selected_game), None)
            if game_data: st.text_area("Game Summary", value=game_data["summary"], height=400)
                
        if st.button("🗑️ Delete Game", type="primary"):
            st.session_state.history = [g for g in st.session_state.history if g["name"] != selected_game]
            with open(HISTORY_FILE, 'w') as f: json.dump(st.session_state.history, f)
            st.rerun()
