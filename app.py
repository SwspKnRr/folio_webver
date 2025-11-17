# app.py
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import yfinance as yf
import pandas as pd

import matplotlib
matplotlib.rcParams['font.family'] = 'Gulim'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.font_manager._rebuild()


from core import (
    run_grid_rebal_backtest,
    compute_signal_stats,
    compute_dd_rsi_2d_stats,
    analyze_vix,
    optimize_param_grid,
    DD_LABELS,
    compute_drawdown,
    compute_rsi,
    analyze_premarket_vs_regular,  # 🔹 새로 추가
)

# ---------- 한글 폰트 설정 (Windows: Gulim) ---------- #
matplotlib.rcParams["font.family"] = "Gulim"
matplotlib.rcParams["axes.unicode_minus"] = False


# ---------- Streamlit 기본 설정 ---------- #
st.set_page_config(page_title="특수 리밸 백테스트 웹앱", layout="wide")

st.title("📈 특수 리밸 + DD/RSI/VIX 심층 분석 툴 (웹 버전)")


# ---------- 세션 상태 초기화 ---------- #
def init_state():
    default_keys = {
        "main_result_df": None,
        "main_trade_log": [],
        "main_final_state": None,
        "main_summary": None,
        "main_params": None,
        "main_signal_stats_df": None,
        "main_signal_horizon": 20,
        "main_prices": None,
        "hc_best_params": None,
    }
    for k, v in default_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------- 공통 함수: 오늘 요약 텍스트 생성 ---------- #
def generate_today_summary(
    prices_series: pd.Series,
    final_state: dict,
    summary: dict,
    params: dict,
    signal_stats_df: pd.DataFrame | None,
    signal_horizon: int,
    fx_rate: float,
    currency: str,
) -> str:
    fs = final_state
    sm = summary
    pm = params

    # 통화 변환
    fx = fx_rate
    if currency == "KRW":
        conv = fx
        unit = "원"
    else:
        conv = 1.0
        unit = "달러"

    total = fs["final_total"] * conv
    cash = fs["final_cash"] * conv
    shares = fs["final_shares"]
    price = fs["last_price"]
    ref_price = fs["ref_price"]

    equity_value = shares * price * conv
    if total <= 0:
        equity_weight = 0.0
        cash_weight = 0.0
    else:
        equity_weight = equity_value / total * 100.0
        cash_weight = cash / total * 100.0

    # 기준가 대비 현재 등락률
    if ref_price > 0:
        diff_pct = (price / ref_price - 1.0) * 100.0
    else:
        diff_pct = 0.0

    up_pct = pm["up_pct"]
    down_pct = pm["down_pct"]
    buy_frac = pm["buy_frac"]
    sell_frac = pm["sell_frac"]
    trade_base_text = pm["trade_base_text"]

    # ---- 룰 기반 오늘 액션 ---- #
    if diff_pct >= up_pct:
        rule_action = "SELL"
        rule_action_text = (
            "지금 가격은 기준가 대비 상승폭이 X_up(상승 트리거)을 넘었습니다 → "
            "룰 기준으로는 '매도(SELL)' 시그널 구간입니다."
        )
    elif diff_pct <= -down_pct:
        rule_action = "BUY"
        rule_action_text = (
            "지금 가격은 기준가 대비 하락폭이 X_down(하락 트리거)을 넘었습니다 → "
            "룰 기준으로는 '매수(BUY)' 시그널 구간입니다."
        )
    else:
        rule_action = "HOLD"
        rule_action_text = (
            "기준가 대비 변동폭이 아직 X_up / X_down 트리거에 도달하지 않았습니다 → "
            "룰 기준으로는 '홀드(HOLD)' 구간입니다."
        )

    # 이론상 기준값
    if pm["trade_base_code"] == "equity":
        base_val_now = equity_value
    else:
        base_val_now = total

    buy_trade_val_theoretical = base_val_now * (buy_frac / 100.0)
    sell_trade_val_theoretical = base_val_now * (sell_frac / 100.0)

    # ---- 과거 통계 기반 관점 ---- #
    hist_action_text = "과거 통계 기반 정보가 아직 없습니다."
    stats_lines = []

    if prices_series is not None and signal_stats_df is not None and not signal_stats_df.empty:
        prices = prices_series.dropna()
        rolling_max = prices.cummax()
        dd_pct_series = (prices / rolling_max - 1.0) * 100.0
        curr_dd_pct = dd_pct_series.iloc[-1]
        if hasattr(curr_dd_pct, "item"):
            curr_dd_pct = curr_dd_pct.item()
        else:
            curr_dd_pct = float(curr_dd_pct)

        bucket_series = pd.Series([curr_dd_pct])
        from core import DD_BINS, DD_LABELS  # 지연 import (순환 방지용)
        bucket_series = pd.cut(
            bucket_series,
            bins=DD_BINS,
            labels=DD_LABELS,
            right=False
        )
        bucket = bucket_series.iloc[0]

        if pd.isna(bucket):
            hist_action_text = "오늘 드로우다운은 정의된 구간 밖이라 통계 기반 추천을 제공하기 어렵습니다."
        else:
            bucket = str(bucket)
            if bucket in signal_stats_df.index:
                row = signal_stats_df.loc[bucket]
                mean_ret = row["mean"]
                median_ret = row["median"]
                count = int(row["count"])
                pos_ratio = row["positive_ratio"]

                stats_lines.append(f"   - 드로우다운 구간: {bucket}")
                stats_lines.append(
                    f"   - 과거 {count}개 사례에서, 앞으로 {signal_horizon}일 평균 수익률: "
                    f"{mean_ret:.2f}% (중앙값 {median_ret:.2f}%)"
                )
                stats_lines.append(f"   - 그 중 플러스가 난 비율: {pos_ratio:.1f}%")

                if mean_ret > 0 and pos_ratio > 55:
                    hist_action_text = (
                        f"과거 통계상, 이런 드로우다운 구간({bucket})에서는 "
                        f"앞으로 {signal_horizon}일 동안 상승하는 경우가 더 많았고 "
                        f"평균 수익률도 플러스였습니다 → "
                        f"통계만 보면 '매수 또는 최소한 유지' 쪽이 유리했던 구간입니다."
                    )
                elif mean_ret < 0 and pos_ratio < 45:
                    hist_action_text = (
                        f"과거 통계상, 이런 드로우다운 구간({bucket})에서는 "
                        f"앞으로 {signal_horizon}일 동안 하락하는 경우가 더 많았고 "
                        f"평균 수익률도 마이너스였습니다 → "
                        f"통계만 보면 '매도 또는 비중 축소' 쪽이 유리했던 구간입니다."
                    )
                else:
                    hist_action_text = (
                        f"과거 통계상, 이런 드로우다운 구간({bucket})에서는 "
                        f"앞으로 {signal_horizon}일 수익률 분포가 비슷하게 섞여 있어 "
                        f"뾰족하게 유리한 방향(매수/매도)이 뚜렷하지 않습니다 → "
                        f"통계만 보면 '중립'에 가깝습니다."
                    )
            else:
                hist_action_text = "해당 드로우다운 구간에 대한 충분한 표본이 없어 통계 기반 추천이 어렵습니다."

    lines = []
    lines.append(f"[티커] {pm['ticker']}  |  리밸 방식: {pm['rebalance_freq']}  |  거래 기준: {trade_base_text}")
    lines.append("")
    lines.append("1) 이 특수 리밸 룰을 과거 전체에 적용했다면?")
    lines.append(f"   - 초기 자산: {sm['initial_capital'] * conv:,.0f} {unit}")
    lines.append(f"   - 전략 최종 자산: {sm['final_total'] * conv:,.0f} {unit} (수익률 {sm['final_return_pct']:.2f}%)")
    lines.append(f"   - Buy&Hold 최종 자산: {sm['buyhold_final'] * conv:,.0f} {unit} (수익률 {sm['buyhold_return_pct']:.2f}%)")
    lines.append(f"   - 전략 CAGR: {sm['cagr_strat_pct']:.2f}% / Buy&Hold CAGR: {sm['cagr_bh_pct']:.2f}%")
    lines.append(f"   - 총 거래 횟수: {sm['num_trades']} 회")
    lines.append("")
    lines.append("2) 백테스트 마지막 날(현재 시점) 기준 포지션")
    lines.append(f"   - 현재 가격: {price:,.2f} USD")
    lines.append(f"   - 기준 가격(ref_price): {ref_price:,.2f} USD")
    lines.append(f"   - 기준가 대비 등락률: {diff_pct:.2f}%")
    lines.append(f"   - 현재 총 자산: {total:,.0f} {unit}")
    lines.append(f"   - 주식 평가액: {equity_value:,.0f} {unit} (비중 {equity_weight:.2f}%)")
    lines.append(f"   - 현금: {cash:,.0f} {unit} (비중 {cash_weight:.2f}%)")
    lines.append("")
    lines.append("3) 특수 리밸 규칙")
    lines.append(f"   - 기준가에서 +{up_pct:.2f}% 오르면: '{trade_base_text}'의 {sell_frac:.2f}% 만큼 '매도'")
    lines.append(f"   - 기준가에서 -{down_pct:.2f}% 내리면: '{trade_base_text}'의 {buy_frac:.2f}% 만큼 '매수'")
    lines.append(f"   - 현재 기준에서 이론상 한 번 매수 시 거래 금액: 약 {buy_trade_val_theoretical:,.0f} {unit}")
    lines.append(f"   - 현재 기준에서 이론상 한 번 매도 시 거래 금액: 약 {sell_trade_val_theoretical:,.0f} {unit}")
    lines.append("")
    lines.append("4) 오늘 기준 추천 해석 (두 관점 결합)")
    lines.append("   ① 룰 기반 관점:")
    lines.append(f"      - {rule_action_text}")
    lines.append("   ② 과거 통계 기반 관점:")
    lines.append(f"      - {hist_action_text}")
    if stats_lines:
        lines.append("      - 참고 통계:")
        lines.extend(stats_lines)
    lines.append("")
    lines.append("※ 정리하면, ①은 '네가 미리 정한 규칙에 충실했을 때 오늘 해야 할 행동',")
    lines.append("   ②는 '과거 데이터만 봤을 때 이런 상황에서 대체로 유리했던 방향(매수/매도/중립)'을 알려주는 정보입니다.")
    lines.append("")
    lines.append("※ 통화 표시만 변환된 것이고, 실제 백테스트 계산은 달러 기준 가격으로 수행됩니다.")

    return "\n".join(lines)


# ---------- 공통 함수: 백테스트 차트 ---------- #
def plot_backtest_chart(result_df: pd.DataFrame, trade_log: list, currency: str, fx_rate: float):
    fx = fx_rate
    if currency == "KRW":
        conv = fx
        unit = "KRW"
    else:
        conv = 1.0
        unit = "USD"

    fig, (ax_price, ax_value) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 가격 라인
    ax_price.plot(result_df.index, result_df["Price"], label="Price")
    ax_price.set_ylabel("가격 (USD)")
    ax_price.set_title("가격 차트")

    # 매수/매도 마커
    for row in trade_log:
        d = row["Date"]
        y = row["Price"]
        if row["Action"] == "BUY":
            marker = "^"
            color = "green"
        else:
            marker = "v"
            color = "red"
        ax_price.scatter(d, y, marker=marker, color=color, s=40, zorder=5)

    ax_price.legend()

    # 자산 곡선
    ax_value.plot(result_df.index, result_df["Strategy_Value"] * conv, label="특수 리밸 전략")
    ax_value.plot(result_df.index, result_df["BuyHold_Value"] * conv, label="Buy&Hold", linestyle="--")
    ax_value.set_ylabel(f"포트폴리오 가치 ({unit})")
    ax_value.set_title("전략 vs Buy&Hold")
    ax_value.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# =========================================================
#                      탭 셋업
# =========================================================
tab_main, tab_hc, tab_ext = st.tabs(["메인 모드", "심층 모드", "프리장 예측 모드"])


# =========================================================
#                      메인 탭
# =========================================================
with tab_main:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("입력값 (메인 특수 리밸)")

        start_min = dt.date(1990, 1, 1)
        today = dt.date.today()

        ticker = st.text_input("티커 (예: VTI, 005930.KS)", "VTI", key="main_ticker")
        start_date = st.date_input(
            "시작일",
            dt.date(2005, 1, 1),
            key="main_start",
            min_value=start_min,
            max_value=today,
        )
        end_date = st.date_input(
            "종료일",
            today,
            key="main_end",
            min_value=start_min,
            max_value=today,
        )

        initial_capital = st.number_input("초기 총 자산 (USD)", value=10000.0, step=1000.0, key="main_init_cap")
        init_weight = st.number_input("초기 주식 비중 (%)", value=50.0, step=5.0, key="main_init_weight")

        up_pct = st.number_input("상승 트리거 X_up (%)", value=5.0, step=1.0, key="main_up_pct")
        down_pct = st.number_input("하락 트리거 X_down (%)", value=5.0, step=1.0, key="main_down_pct")
        buy_frac = st.number_input("하락 시 매수 비중 (%)", value=10.0, step=1.0, key="main_buy_frac")
        sell_frac = st.number_input("상승 시 매도 비중 (%)", value=10.0, step=1.0, key="main_sell_frac")

        freq_text = st.selectbox(
            "리밸 방식 / 주기",
            ["트리거 즉시", "1일 주기", "1주 주기", "1개월 주기", "4개월 주기"],
            index=0,
            key="main_freq_text",
        )

        trade_base_text = st.selectbox(
            "거래 기준 (Y% 계산 기준)",
            ["포트폴리오 기준", "주식 평가액 기준"],
            index=0,
            key="main_trade_base_text",
        )

        horizon_days = st.number_input("통계용 미래 기간 N (일)", value=20, step=5, key="main_horizon")

        st.markdown("---")
        st.subheader("표시 통화 설정")
        currency = st.radio("표시 통화", ["USD", "KRW"], index=0, key="main_currency")
        fx_rate = st.number_input("환율 (1 USD = ? KRW)", value=1400.0, step=10.0, key="main_fx")

        st.markdown("---")
        run_bt = st.button("백테스트 실행", key="btn_run_bt")
        run_opt = st.button("최적 파라미터 탐색", key="btn_opt")

    with col_right:
        st.subheader("요약 / 오늘 액션")

        # --- 백테스트 실행 버튼 ---
        if run_bt:
            if not ticker:
                st.error("티커를 입력하세요.")
            else:
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                try:
                    df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True)
                except Exception as e:
                    st.error(f"데이터 다운로드 실패: {e}")
                    df = None

                if df is None or df.empty or "Close" not in df.columns:
                    st.error("다운로드된 데이터가 없습니다. 티커/기간을 다시 확인하세요.")
                else:
                    # 실제로 받은 데이터 기간 보여주기
                    try:
                        st.caption(
                            f"📅 다운로드된 실제 데이터 기간: "
                            f"{df.index.min().date()} ~ {df.index.max().date()}"
                        )
                    except Exception:
                        pass

                    prices = df["Close"].dropna()
                    if prices.empty:
                        st.error("종가 데이터가 비어 있습니다.")
                    else:
                        freq_map = {
                            "트리거 즉시": "event",
                            "1일 주기": "1D",
                            "1주 주기": "1W",
                            "1개월 주기": "1M",
                            "4개월 주기": "4M",
                        }
                        freq_code = freq_map.get(freq_text, "event")
                        trade_base_code = "equity" if "주식" in trade_base_text else "total"

                        try:
                            result_df, trade_log, final_state, summary = run_grid_rebal_backtest(
                                prices,
                                initial_capital,
                                init_weight,
                                up_pct,
                                down_pct,
                                buy_frac,
                                sell_frac,
                                trade_base=trade_base_code,
                                rebalance_freq=freq_code,
                            )
                        except Exception as e:
                            st.error(f"백테스트 중 오류: {e}")
                        else:
                            # 드로우다운 기반 통계
                            try:
                                signal_stats_df = compute_signal_stats(prices, int(horizon_days))
                            except Exception:
                                signal_stats_df = None

                            st.session_state["main_result_df"] = result_df
                            st.session_state["main_trade_log"] = trade_log
                            st.session_state["main_final_state"] = final_state
                            st.session_state["main_summary"] = summary
                            st.session_state["main_params"] = {
                                "ticker": ticker,
                                "up_pct": up_pct,
                                "down_pct": down_pct,
                                "buy_frac": buy_frac,
                                "sell_frac": sell_frac,
                                "init_weight": init_weight,
                                "rebalance_freq": freq_text,
                                "trade_base_code": trade_base_code,
                                "trade_base_text": trade_base_text,
                            }
                            st.session_state["main_signal_stats_df"] = signal_stats_df
                            st.session_state["main_signal_horizon"] = int(horizon_days)
                            st.session_state["main_prices"] = prices

                            summary_text = generate_today_summary(
                                prices_series=prices,
                                final_state=final_state,
                                summary=summary,
                                params=st.session_state["main_params"],
                                signal_stats_df=signal_stats_df,
                                signal_horizon=int(horizon_days),
                                fx_rate=fx_rate,
                                currency=currency,
                            )
                            st.text(summary_text)

                            st.success("백테스트 완료!")

                            # 차트
                            fig = plot_backtest_chart(result_df, trade_log, currency, fx_rate)
                            st.pyplot(fig)

                            # 거래 로그 테이블
                            if trade_log:
                                st.markdown("### 거래 로그")
                                st.dataframe(pd.DataFrame(trade_log))

        # 이미 실행된 게 있으면 그대로 보여주기
        elif (
            st.session_state["main_result_df"] is not None
            and st.session_state["main_final_state"] is not None
            and st.session_state["main_summary"] is not None
            and st.session_state["main_params"] is not None
        ):
            summary_text = generate_today_summary(
                prices_series=st.session_state["main_prices"],
                final_state=st.session_state["main_final_state"],
                summary=st.session_state["main_summary"],
                params=st.session_state["main_params"],
                signal_stats_df=st.session_state["main_signal_stats_df"],
                signal_horizon=st.session_state["main_signal_horizon"],
                fx_rate=fx_rate,
                currency=currency,
            )
            st.text(summary_text)

            fig = plot_backtest_chart(
                st.session_state["main_result_df"],
                st.session_state["main_trade_log"],
                currency,
                fx_rate,
            )
            st.pyplot(fig)

            if st.session_state["main_trade_log"]:
                st.markdown("### 거래 로그")
                st.dataframe(pd.DataFrame(st.session_state["main_trade_log"]))

        # --- 최적 파라미터 탐색 버튼 ---
        if run_opt:
            if st.session_state["main_prices"] is None:
                st.info("먼저 백테스트를 한 번 실행해서 데이터(가격 시계열)를 불러와 주세요.")
            else:
                prices = st.session_state["main_prices"]

                freq_map = {
                    "트리거 즉시": "event",
                    "1일 주기": "1D",
                    "1주 주기": "1W",
                    "1개월 주기": "1M",
                    "4개월 주기": "4M",
                }
                freq_code = freq_map.get(freq_text, "event")
                trade_base_code = "equity" if "주식" in trade_base_text else "total"

                best_combo, total_tests, success_tests = optimize_param_grid(
                    prices,
                    initial_capital,
                    init_weight,
                    freq_code,
                    trade_base_code,
                )

                if best_combo is None:
                    st.error("파라미터 탐색 중 유효한 조합이 하나도 나오지 않았습니다.")
                else:
                    d_best, u_best, bf_best, sf_best, summary_best = best_combo

                    # 통화 변환
                    if currency == "KRW":
                        conv = fx_rate
                        unit = "원"
                    else:
                        conv = 1.0
                        unit = "달러"

                    init_cap_conv = summary_best["initial_capital"] * conv
                    final_conv = summary_best["final_total"] * conv
                    bh_final_conv = summary_best["buyhold_final"] * conv

                    lines = []
                    lines.append("")
                    lines.append(f"[최적 파라미터 탐색 결과] (티커: {ticker}, 리밸 방식: {freq_text}, 거래 기준: {trade_base_text})")
                    lines.append(f" - 테스트한 조합 수: {total_tests}개 (성공 {success_tests}개)")
                    lines.append("")
                    lines.append(
                        f" ▶ 이 구간 데이터 기준, '{trade_base_text}' 기준으로 "
                        f"{d_best}% 하락할 때마다 {bf_best}% 매수하고, "
                        f"{u_best}% 상승할 때마다 {sf_best}% 매도했을 때 "
                        f"가장 높은 최종 자산이 나왔습니다."
                    )
                    lines.append("")
                    lines.append(f"   - 초기 자산: {init_cap_conv:,.0f} {unit}")
                    lines.append(f"   - 전략 최종 자산: {final_conv:,.0f} {unit} (수익률 {summary_best['final_return_pct']:.2f}%)")
                    lines.append(
                        f"   - 동일 구간 Buy&Hold 최종 자산: {bh_final_conv:,.0f} {unit} "
                        f"(수익률 {summary_best['buyhold_return_pct']:.2f}%)"
                    )
                    lines.append(
                        f"   - 전략 CAGR: {summary_best['cagr_strat_pct']:.2f}% / "
                        f"Buy&Hold CAGR: {summary_best['cagr_bh_pct']:.2f}%"
                    )
                    lines.append(f"   - 총 거래 횟수: {summary_best['num_trades']} 회")
                    lines.append("")
                    lines.append("※ 이 최적 파라미터는 '지금 설정한 기간'과 '리밸 주기 / 거래 기준'을 고정해 놓고,")
                    lines.append("   여러 조합을 돌려 본 결과이므로, 다른 기간/주기로 바꾸면 최적값도 달라질 수 있습니다.")
                    lines.append("")
                    lines.append("※ 아래 추천 값은 참고해서 왼쪽 입력칸에 직접 입력해도 됩니다.")
                    lines.append(f"   · 추천 X_down: {d_best}")
                    lines.append(f"   · 추천 X_up: {u_best}")
                    lines.append(f"   · 추천 매수 비중: {bf_best}")
                    lines.append(f"   · 추천 매도 비중: {sf_best}")

                    st.text("\n".join(lines))


# =========================================================
#                      심층 모드 탭
# =========================================================
with tab_hc:
    st.subheader("심층 모드 (DD / RSI / VIX / F&G + 최적 파라미터)")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        start_min = dt.date(1990, 1, 1)
        today = dt.date.today()

        ticker_hc = st.text_input("티커", "VTI", key="hc_ticker")
        start_hc = st.date_input(
            "시작일 (YYYY-MM-DD)",
            dt.date(2005, 1, 1),
            key="hc_start",
            min_value=start_min,
            max_value=today,
        )
        end_hc = st.date_input(
            "종료일 (YYYY-MM-DD)",
            today,
            key="hc_end",
            min_value=start_min,
            max_value=today,
        )

        rsi_period = st.number_input("RSI 기간", value=14, step=1, key="hc_rsi_period")
        horizon_hc = st.number_input("미래 N일 (수익률)", value=20, step=5, key="hc_horizon")

        fg_value_str = st.text_input("공포·탐욕지수 (0~100, 수동입력)", "50", key="hc_fg")

        run_hc = st.button("심층 분석 실행", key="btn_run_hc")

    with col_r:
        if run_hc:
            if not ticker_hc:
                st.error("티커를 입력하세요.")
            else:
                start_str = start_hc.strftime("%Y-%m-%d")
                end_str = end_hc.strftime("%Y-%m-%d")

                try:
                    data = yf.download(ticker_hc, start=start_str, end=end_str, progress=False, auto_adjust=True)
                except Exception as e:
                    st.error(f"데이터 다운로드 실패: {e}")
                    data = None

                if data is None or data.empty or "Close" not in data.columns:
                    st.error("데이터가 비어 있습니다.")
                else:
                    # 실제로 받은 데이터 기간 보여주기
                    try:
                        st.caption(
                            f"📅 다운로드된 실제 데이터 기간: "
                            f"{data.index.min().date()} ~ {data.index.max().date()}"
                        )
                    except Exception:
                        pass

                    close = data["Close"].dropna()
                    if close.empty or len(close) < 60:
                        st.error("데이터가 너무 적습니다.")
                    else:
                        dd = compute_drawdown(close)
                        rsi = compute_rsi(close, int(rsi_period))

                        today_dd = dd.iloc[-1]
                        today_dd = float(today_dd.item() if hasattr(today_dd, "item") else today_dd)

                        today_rsi = rsi.iloc[-1]
                        today_rsi = float(today_rsi.item() if hasattr(today_rsi, "item") else today_rsi)

                        # RSI 버킷
                        if today_rsi < 30:
                            rsi_bucket = "RSI<30"
                        elif today_rsi < 50:
                            rsi_bucket = "30~50"
                        elif today_rsi < 70:
                            rsi_bucket = "50~70"
                        else:
                            rsi_bucket = ">=70"

                        # DD 버킷
                        if today_dd < -40:
                            dd_bucket = "< -40%"
                        elif today_dd < -30:
                            dd_bucket = "-40 ~ -30%"
                        elif today_dd < -20:
                            dd_bucket = "-30 ~ -20%"
                        elif today_dd < -10:
                            dd_bucket = "-20 ~ -10%"
                        elif today_dd < -5:
                            dd_bucket = "-10 ~ -5%"
                        elif today_dd < 0:
                            dd_bucket = "-5 ~ 0%"
                        elif today_dd < 5:
                            dd_bucket = "0 ~ +5%"
                        elif today_dd < 10:
                            dd_bucket = "+5 ~ +10%"
                        elif today_dd < 20:
                            dd_bucket = "+10 ~ +20%"
                        else:
                            dd_bucket = ">= +20%"

                        # VIX
                        vix_now, vix_pct, vix_series = analyze_vix(start_str, end_str)

                        # F&G
                        try:
                            fg_value = float(fg_value_str)
                        except Exception:
                            fg_value = None

                        st.markdown("### 1) 오늘 DD / RSI / VIX / F&G 상태")

                        status_lines = []
                        status_lines.append(f"[{ticker_hc}] 심층 분석 결과")
                        status_lines.append("")
                        status_lines.append(f"오늘 드로우다운 DD: {today_dd:.2f}%  ({dd_bucket})")
                        status_lines.append(f"오늘 RSI: {today_rsi:.2f}  ({rsi_bucket})")
                        status_lines.append("")

                        if vix_now is not None:
                            status_lines.append(f"오늘 VIX: {vix_now:.2f}")

                            vix_pct_val = None
                            try:
                                if vix_pct is not None:
                                    if isinstance(vix_pct, pd.Series):
                                        vix_pct_val = float(vix_pct.iloc[-1])
                                    else:
                                        vix_pct_val = float(vix_pct)
                            except Exception as e:
                                status_lines.append(f"VIX 분위수 변환 중 오류: {e}")

                            if vix_pct_val is not None:
                                status_lines.append(f"과거 대비 분위수: 상위 {vix_pct_val:.1f}%")
                            else:
                                status_lines.append("과거 대비 분위수: 계산 불가 (데이터 형식 문제 / 표본 부족)")

                            if vix_now < 15:
                                status_lines.append("VIX 해석: 매우 안정적인 시장 상황.")
                            elif vix_now < 25:
                                status_lines.append("VIX 해석: 정상 범위의 변동성.")
                            elif vix_now < 40:
                                status_lines.append("VIX 해석: 시장이 긴장된 상태. 리스크 관리 필요.")
                            else:
                                status_lines.append("VIX 해석: 공포장 수준. 변동성 매우 큼.")
                            status_lines.append("")
                        else:
                            status_lines.append("VIX 데이터 없음.")
                            status_lines.append("")

                        if fg_value is not None:
                            status_lines.append(f"F&G Index (수동 입력): {fg_value:.1f}")
                            if fg_value < 25:
                                status_lines.append("F&G 해석: 극단적 공포 구간.")
                            elif fg_value < 45:
                                status_lines.append("F&G 해석: 공포 구간.")
                            elif fg_value < 55:
                                status_lines.append("F&G 해석: 중립.")
                            elif fg_value < 75:
                                status_lines.append("F&G 해석: 탐욕.")
                            else:
                                status_lines.append("F&G 해석: 극단적 탐욕.")
                        else:
                            status_lines.append("F&G 값 제공되지 않음.")

                        st.text("\n".join(status_lines))

                        # 2D 통계
                        st.markdown("### 2) DD × RSI 2D 통계 (셀: 평균 N일 수익률 %)")

                        pivot_mean, raw_df = compute_dd_rsi_2d_stats(close, int(rsi_period), int(horizon_hc))

                        if pivot_mean is None or pivot_mean.empty:
                            st.info("2D DD×RSI 통계를 계산하기에 데이터가 부족합니다.")
                        else:
                            # 보기 좋게 정렬
                            rsi_cols = ["RSI<30", "30~50", "50~70", ">=70"]
                            pivot_display = pivot_mean.reindex(index=DD_LABELS, columns=rsi_cols)
                            st.dataframe(pivot_display)

                            # 오늘 구간의 과거 평균 수익률 / 승률
                            today_mask = (raw_df["DD_Bucket"] == dd_bucket) & (raw_df["RSI_Bucket"] == rsi_bucket)
                            subset = raw_df.loc[today_mask]

                            if subset.empty:
                                hc_comment = (
                                    f"오늘 위치는 DD={dd_bucket}, RSI={rsi_bucket} 이지만, "
                                    f"이 조합에 해당하는 과거 표본이 거의 없습니다 → "
                                    f"통계만으로 방향성 판단이 어렵습니다."
                                )
                            else:
                                mean_ret = subset["FwdRet_Pct"].mean()
                                pos_ratio = (subset["FwdRet_Pct"] > 0).mean() * 100.0
                                count = len(subset)

                                if mean_ret > 0 and pos_ratio > 55:
                                    core_view = "매수/비중 확대 쪽이 통계적으로 유리했던 구간"
                                    core_action = "공격적 매수 또는 최소한 홀드 쪽에 무게를 둘 수 있는 구간."
                                elif mean_ret < 0 and pos_ratio < 45:
                                    core_view = "매도/비중 축소 쪽이 통계적으로 유리했던 구간"
                                    core_action = "익절/손절을 통해 비중을 줄이는 쪽을 우선 고려할 만한 구간."
                                else:
                                    core_view = "방향성이 뚜렷하지 않은 중립 구간"
                                    core_action = "무리한 매매보다 점진적 분할 매수·매도 or 관망이 더 적절한 구간."

                                hc_comment = (
                                    f"오늘 위치: DD={dd_bucket}, RSI={rsi_bucket}\n"
                                    f"- 이 조합에 해당하는 과거 표본 수: {count}개\n"
                                    f"- 향후 {int(horizon_hc)}일 평균 수익률: {mean_ret:.2f}%\n"
                                    f"- 플러스 비율: {pos_ratio:.1f}%\n"
                                    f"→ 통계상 해석: {core_view}\n"
                                    f"→ 액션 힌트: {core_action}"
                                )

                            st.markdown("### 3) 통계 기반 해석")
                            st.text(hc_comment)

                        # 메인 탭 설정 기반 최적 파라미터 (같은 기간에 대해)
                        st.markdown("### 4) 동일 구간 특수 리밸 최적 파라미터 탐색")

                        # 메인 탭의 초기 자산 / 비중 / 주기 / 거래 기준 사용
                        try:
                            initial_capital_main = float(st.session_state.get("main_init_cap", 10000.0))
                            init_weight_main = float(st.session_state.get("main_init_weight", 50.0))
                        except Exception:
                            initial_capital_main = 10000.0
                            init_weight_main = 50.0

                        freq_text_main = st.session_state.get("main_freq_text", "트리거 즉시")
                        trade_base_text_main = st.session_state.get("main_trade_base_text", "포트폴리오 기준")

                        freq_map = {
                            "트리거 즉시": "event",
                            "1일 주기": "1D",
                            "1주 주기": "1W",
                            "1개월 주기": "1M",
                            "4개월 주기": "4M",
                        }
                        freq_code_main = freq_map.get(freq_text_main, "event")
                        trade_base_code_main = "equity" if "주식" in trade_base_text_main else "total"

                        best_combo, total_tests, success_tests = optimize_param_grid(
                            close,
                            initial_capital_main,
                            init_weight_main,
                            freq_code_main,
                            trade_base_code_main,
                        )

                        if best_combo is None:
                            st.info("심층 모드에서 파라미터 탐색 실패 (유효 조합 없음).")
                            st.session_state["hc_best_params"] = None
                        else:
                            d_best, u_best, bf_best, sf_best, summary_best = best_combo

                            # 통화 변환 (메인 탭 기준)
                            main_currency = st.session_state.get("main_currency", "USD")
                            main_fx = st.session_state.get("main_fx", 1400.0)
                            if main_currency == "KRW":
                                conv = main_fx
                                unit = "원"
                            else:
                                conv = 1.0
                                unit = "달러"

                            init_cap_conv = summary_best["initial_capital"] * conv
                            final_conv = summary_best["final_total"] * conv
                            bh_final_conv = summary_best["buyhold_final"] * conv

                            lines = []
                            lines.append("【2단계】 동일 구간에서 특수 리밸 파라미터 그리드 탐색 완료.")
                            lines.append(f" - 테스트된 조합: {total_tests}개 (성공 {success_tests}개)")
                            lines.append(
                                f" - '{trade_base_text_main}' 기준으로 "
                                f"{d_best}% 하락마다 {bf_best}% 매수, "
                                f"{u_best}% 상승마다 {sf_best}% 매도했을 때"
                            )
                            lines.append("   가장 높은 최종 자산이 나왔습니다.")
                            lines.append(f"   · 초기 자산: {init_cap_conv:,.0f} {unit}")
                            lines.append(
                                f"   · 전략 최종 자산: {final_conv:,.0f} {unit} "
                                f"(수익률 {summary_best['final_return_pct']:.2f}%)"
                            )
                            lines.append(
                                f"   · 동일 구간 Buy&Hold: {bh_final_conv:,.0f} {unit} "
                                f"(수익률 {summary_best['buyhold_return_pct']:.2f}%)"
                            )
                            lines.append(
                                f"   · 전략 CAGR: {summary_best['cagr_strat_pct']:.2f}% / "
                                f"Buy&Hold CAGR: {summary_best['cagr_bh_pct']:.2f}%"
                            )
                            lines.append(f"   · 총 거래 횟수: {summary_best['num_trades']}회")
                            lines.append("")
                            lines.append(
                                "※ 이제 아래 추천 값을 참고해서, 메인 탭 입력값을 수동으로 조정할 수 있습니다."
                            )
                            lines.append(f"   · 추천 X_down: {d_best}")
                            lines.append(f"   · 추천 X_up: {u_best}")
                            lines.append(f"   · 추천 매수 비중: {bf_best}")
                            lines.append(f"   · 추천 매도 비중: {sf_best}")

                            st.text("\n".join(lines))

                            st.session_state["hc_best_params"] = {
                                "ticker": ticker_hc,
                                "start": start_str,
                                "end": end_str,
                                "down": d_best,
                                "up": u_best,
                                "buy_frac": bf_best,
                                "sell_frac": sf_best,
                                "freq_text": freq_text_main,
                                "freq_code": freq_code_main,
                                "trade_base_code": trade_base_code_main,
                                "trade_base_text": trade_base_text_main,
                            }

        else:
            st.info("왼쪽에서 설정 입력 후 '심층 분석 실행'을 눌러주세요.")


# =========================================================
#                 프리장 예측 모드 탭
# =========================================================
with tab_ext:
    st.subheader("프리장 vs 본장 변동률 기반 예측 모드 (인트라데이 / Extended Hours)")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        today = dt.date.today()
        max_span_days = 59  # yfinance 인트라데이 30m 기준 최대 약 60일
        start_min_ext = today - dt.timedelta(days=max_span_days)

        ticker_ext = st.text_input("티커 (미국 ETF/주식 권장, 예: QQQ, SPY, VTI)", "QQQ", key="ext_ticker")

        start_ext = st.date_input(
            "시작일 (최대 약 60일 전까지)",
            today - dt.timedelta(days=30),
            key="ext_start",
            min_value=start_min_ext,
            max_value=today,
        )
        end_ext = st.date_input(
            "종료일",
            today,
            key="ext_end",
            min_value=start_min_ext,
            max_value=today,
        )

        interval_label = st.selectbox(
            "인트라데이 캔들 간격 (프리장/본장 구분용)",
            ["30분", "60분"],
            index=0,
            key="ext_interval",
        )
        interval_map = {"30분": "30m", "60분": "60m"}
        interval_code = interval_map[interval_label]

        strong_thr = st.number_input(
            "강한 프리장 변동 기준 (절댓값, %)",
            value=1.0,
            step=0.5,
            key="ext_strong_thr",
        )

        st.caption(
            "- 프리장: 04:00 ~ 09:30 (미국 동부시간 기준)\n"
            "- 본장: 09:30 ~ 16:00 (미국 동부시간 기준)\n"
            "- yfinance 인트라데이 제한 때문에 최대 약 60일만 분석 가능"
        )

        run_ext = st.button("프리장 vs 본장 통계 분석 실행", key="btn_run_ext")

    with col_r:
        if run_ext:
            if not ticker_ext:
                st.error("티커를 입력하세요.")
            else:
                if start_ext > end_ext:
                    st.error("시작일이 종료일보다 이후입니다.")
                else:
                    start_str = start_ext.strftime("%Y-%m-%d")
                    end_str = (end_ext + dt.timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end는 exclusive 느낌이라 하루 더

                    try:
                        daily_df, stats = analyze_premarket_vs_regular(
                            ticker_ext,
                            start=start_str,
                            end=end_str,
                            interval=interval_code,
                            strong_move_thresh=float(strong_thr),
                        )
                    except Exception as e:
                        st.error(f"프리장/본장 분석 중 오류: {e}")
                    else:
                        n = stats["n_days"]
                        st.markdown("### 1) 사용된 표본 개요")

                        info_lines = []
                        info_lines.append(f"[{ticker_ext}] 인트라데이 데이터 기반 프리장 vs 본장 분석")
                        info_lines.append("")
                        info_lines.append(
                            f"- 실제 분석된 거래일 수: {n}일 "
                            f"({stats['start_date']} ~ {stats['end_date']})"
                        )
                        info_lines.append(f"- 인트라데이 간격: {interval_label} 캔들")
                        info_lines.append("")

                        corr = stats["corr_pre_reg"]
                        slope = stats["reg_slope_per_1pct_pre"]

                        if corr is not None:
                            info_lines.append(f"- 프리장 수익률 vs 본장 수익률 피어슨 상관계수: {corr:.3f}")
                        else:
                            info_lines.append("- 상관계수: 계산 불가 (표본 변동성 부족)")

                        if slope is not None:
                            info_lines.append(
                                f"- 단순 회귀 기울기: 프리장 1% 변화 → 본장 평균 {slope:.2f}% 변화"
                            )
                        else:
                            info_lines.append("- 회귀 기울기: 계산 불가 (분산 0에 가까움)")

                        st.text("\n".join(info_lines))

                        st.markdown("### 2) 조건부 확률 / 패턴 통계")

                        def fmt(v):
                            if v is None:
                                return "계산 불가"
                            return f"{v:.1f}%"

                        p_up = fmt(stats["p_reg_up"])
                        p_down = fmt(stats["p_reg_down"])
                        p_up_pre_up = fmt(stats["p_reg_up_given_pre_up"])
                        p_up_pre_down = fmt(stats["p_reg_up_given_pre_down"])
                        p_same_sign = fmt(stats["p_same_sign"])
                        thr = stats["strong_move_thresh"]
                        p_up_strong_up = fmt(stats["p_reg_up_given_strong_pre_up"])
                        p_up_strong_down = fmt(stats["p_reg_up_given_strong_pre_down"])

                        cond_lines = []
                        cond_lines.append(f"- 전체 본장 상승 확률: {p_up} (하락: {p_down})")
                        cond_lines.append("")
                        cond_lines.append(f"- 프리장 상승(>0%)인 날, 본장 상승 확률: {p_up_pre_up}")
                        cond_lines.append(f"- 프리장 하락(<0%)인 날, 본장 상승 확률: {p_up_pre_down}")
                        cond_lines.append("")
                        cond_lines.append(f"- 프리장과 본장이 같은 방향(둘 다 상승/둘 다 하락)일 확률: {p_same_sign}")
                        cond_lines.append("")
                        cond_lines.append(
                            f"- |프리장| ≥ {thr:.1f}% 인 '강한 프리장'일 때:"
                        )
                        cond_lines.append(
                            f"   · 프리장이 +{thr:.1f}% 이상일 때 본장 상승 확률: {p_up_strong_up}"
                        )
                        cond_lines.append(
                            f"   · 프리장이 -{thr:.1f}% 이하일 때 본장 상승 확률: {p_up_strong_down}"
                        )

                        st.text("\n".join(cond_lines))

                        st.markdown("### 3) 일별 프리장 / 본장 수익률 테이블 (최근 30일)")

                        show_df = daily_df[["PreRet", "RegRet"]].copy()
                        show_df.columns = ["프리장 수익률(%)", "본장 수익률(%)"]
                        st.dataframe(
                            show_df.sort_index(ascending=False).head(30)
                        )

                        st.markdown("### 4) 프리장 vs 본장 수익률 산점도")

                        fig_scat, ax_scat = plt.subplots(figsize=(5, 4))
                        ax_scat.scatter(daily_df["PreRet"], daily_df["RegRet"], alpha=0.7)
                        ax_scat.axhline(0, linestyle="--", linewidth=1)
                        ax_scat.axvline(0, linestyle="--", linewidth=1)
                        ax_scat.set_xlabel("프리장 수익률 (%)")
                        ax_scat.set_ylabel("본장 수익률 (%)")
                        ax_scat.set_title(f"{ticker_ext} 프리장 vs 본장 일별 수익률 산점도")
                        fig_scat.tight_layout()
                        st.pyplot(fig_scat)

                        st.info(
                            "※ 이 탭은 어디까지나 '통계적 경향'을 보여주는 용도고, "
                            "단기 예측을 과신하면 안 됨. "
                            "프리장 방향과 강도, 본장 실제 방향이 어떻게 엮이는지 '감각 보정'용으로 쓰면 됨."
                        )
