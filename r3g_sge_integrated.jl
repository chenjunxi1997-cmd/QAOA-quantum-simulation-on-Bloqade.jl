#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#  Q3_sge.jl  —  Q3 Hypercube QAOA MaxCut · GSM+CSM Bayesian Optimisation
#  Scheduler Support: SGE / SLURM / PBS / Local
#  Structure follows parallel_computing_template.jl exactly.
#
#  HOW TO RUN:
#    julia Q3_sge.jl                              # full Phase 1+2 sweep
#    julia Q3_sge.jl pscale_chain [d_init] [step] [n_grid]  # adaptive p=1→5 chain
#    julia Q3_sge.jl pscale <p> [d_fixed]         # single-p BO at fixed d
#    julia Q3_sge.jl baseline <method> [p] [d]    # classical optimiser comparison
#    julia Q3_sge.jl joint                         # joint (d,γ,β) 7D BO
#    julia Q3_sge.jl verify                        # edge-count sanity check
#    julia Q3_sge.jl aggregate <results_dir> [n_bo_iters]   # Phase 2 only
#
#  Julia 1.11+ compatibility fixes applied:
#    1. `something(bθ, fallback)` → `isnothing(bθ) ? fallback : bθ`
#    2. `repeat(T[0.0,0.0], p)` → `zeros(T, 2*p)`
#    3. All @printf calls use paren form: @printf("fmt", args)
#
#  Task mapping (120 tasks = 6 d-values × 20 seeds):
#    task   1-20:  d=11.5 μm  seeds 1..20
#    task  21-40:  d=12.0 μm  seeds 1..20
#    task  41-60:  d=12.5 μm  seeds 1..20
#    task  61-80:  d=13.0 μm  seeds 1..20
#    task  81-100: d=13.5 μm  seeds 1..20
#    task 101-120: d=14.0 μm  seeds 1..20
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%

# ── Verify mode: skip all heavy imports ─────────────────────────────────────
if length(ARGS) >= 1 && ARGS[1] == "verify"
    println("Edge verification mode — skipping package load")
    # (verify block at bottom of file)
end

# ── CondaPkg lock prevention ───────────────────────────────────────────────
# Must be set BEFORE any `using Bloqade` / PythonCall on the master process.
# This tells CondaPkg "Python env already exists, skip resolution and lock".
# Works whether launched from CMD, Jupyter run(), or cluster scheduler.
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using Pkg
Pkg.activate("ChenQAOA")          # adjust to your project name or "." for current dir

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── Core packages: load on MAIN process BEFORE addprocs ─────────────────────
# Types used in inter-process communication must be loaded here.
using Distributed
using Dates
using Printf
using Serialization
using Statistics
using LinearAlgebra

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── GLOBAL PARAMETERS ───────────────────────────────────────────────────────
const D_CANDIDATES = [12.5, 13.0, 13.5, 14.0, 14.5, 15.0]  # shifted right again: best ar at d=14.0, exploring up to 15.0 μm
const SEEDS        = collect(1:20)                      # 20 seeds for statistical robustness (publication-grade)
const N_TASKS      = length(D_CANDIDATES) * length(SEEDS)  # Phase-1 tasks
const N_JOINT_TASKS = length(SEEDS)                         # one run per seed
const N_BO_ITER_D  = 16                                 # Phase-2 spacing BO steps
const QAOA_BUDGET  = 500                                # BO budget per (d,seed)
const QAOA_N_INIT  = 40                                 # Halton warm-start points


#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── CLUSTER DETECTION & WORKER LAUNCH ───────────────────────────────────────
# Follows the template exactly: SGE → SLURM → PBS → local fallback.

# Spawn workers for all compute-heavy modes
_compute_modes = Set(["", "aggregate", "pscale", "pscale_chain",
                      "baseline", "joint",
                      "gnn_qaoa_weighted", "gnn_pretrain_and_qaoa_weighted",
                      "gnn_batch_qaoa_weighted", "kernel_ablation", "landscape_scan",
                      "transfer_bo", "omega_scan", "noise_eval"])
_current_mode  = length(ARGS) >= 1 ? ARGS[1] : ""

# ══════════════════════════════════════════════════════════════════════════
# 日志分流: 完整日志写文件, 控制台只放摘要
# ──────────────────────────────────────────────────────────────────────────
# 为什么这么设计: Jupyter 的输出上限是 512KB, 一超就 "Excessive output
# truncated", 后面的内容(包括最后那几张论文数据表)全看不到。但如果干脆把
# 全部输出重定向到文件, notebook 里又完全没有反馈, 跟卡死没区别。
# 所以做成分流 —— 每一行都完整写进文件, 但只有**重要的**行同时打到控制台。
#
# 重要性分级(用 log_note 的第二个参数):
#   :always  始终显示 —— 表头、任务开始/完成、最终汇总表、错误
#   :summary 摘要级   —— 每个 (p, condition) 跑完的结果行、心跳
#   :detail  仅入文件 —— BO 每次迭代、精修命中等高频细节
# 控制台默认显示 :always + :summary, 用 JULIA_CONSOLE_LEVEL=always 可以再收紧。
#
# 用法: 设 JULIA_LOG_FILE=路径 即可启用。不设的话行为跟以前完全一样。
# ══════════════════════════════════════════════════════════════════════════
const LOG_FILE_IO  = Ref{Union{Nothing,IO}}(nothing)
const CONSOLE_MIN  = Ref(:summary)      # :always 更安静, :detail 全打(会超限)
const _LOG_RANK    = Dict(:always=>3, :summary=>2, :detail=>1)
const LOG_LINES    = Ref(0)
const CONSOLE_LINES= Ref(0)

"""把一行同时写日志文件和(按级别)控制台。
   level: :always / :summary / :detail —— 见上面的分级说明。
   [用法] 主进程用这个; worker 侧用 @everywhere 块里的 wlog(两者不能同名,
   否则 @everywhere 在主进程执行时会把这个覆盖掉, 导致日志文件为空)。"""
function log_note(s::AbstractString, level::Symbol=:summary)
    LOG_LINES[] += 1
    io = LOG_FILE_IO[]
    io !== nothing && (println(io, s); LOG_LINES[] % 50 == 0 && flush(io))
    if _LOG_RANK[level] >= _LOG_RANK[CONSOLE_MIN[]]
        println(s); flush(stdout)
        CONSOLE_LINES[] += 1
    end
    return nothing
end

# [说明] 这里**不能**用自定义 IO 包一层 tee 再 redirect_stdout ——
# redirect_stdout 只接受 Pipe/IOStream/DevNull 这几种具体类型, 传自定义 IO
# 会报 MethodError: no method matching (::Base.RedirectStdStream)(::TeeIO)。
# 改成: log_note 自己决定写哪儿(控制台/文件/两者), worker 侧的日志行随返回值
# 带回主进程, 由主进程统一落盘。

# 主进程的普通 println(表格、诊断等)也要进日志文件。用一个宏替换代价太大,
# 这里改成: 主进程用 mprintln 代替 println —— 它等价于 log_note(s, :always),
# 即"控制台和文件都写"。批量模式里量大的输出已经走 log_note/wlog 分级了,
# 剩下的表格类输出本来就该两边都有。
mprintln(args...) = log_note(sprint(print, args...), :always)

if haskey(ENV, "JULIA_LOG_FILE") && _current_mode != ""
    _logpath = ENV["JULIA_LOG_FILE"]
    LOG_FILE_IO[] = open(_logpath, "w")
    lvl = get(ENV, "JULIA_CONSOLE_LEVEL", "summary")
    CONSOLE_MIN[] = lvl == "always" ? :always : (lvl == "detail" ? :detail : :summary)
    println("完整日志 → $(_logpath)")
    println("控制台只显示 $(CONSOLE_MIN[]) 级别及以上 (BO每步细节全在文件里)")
    println("想更安静: JULIA_CONSOLE_LEVEL=always;  想全打(会超Jupyter上限): =detail")
    flush(stdout)
    # 环境变量传给 worker(addprocs 会继承), 让 worker 侧按同样的级别过滤
    ENV["JULIA_CONSOLE_LEVEL"] = string(CONSOLE_MIN[])
end

if _current_mode in _compute_modes

    # ══════════════════════════════════════════════════════════════════════
    # 核数检测
    # ──────────────────────────────────────────────────────────────────────
    # [重要] 在有调度器的集群上, 能用多少核由**调度器的分配**决定, 不是"全集群
    # 还剩多少空闲核"。cgroups/cpuset 通常会硬性限制到分配额度, 强行 spawn 更多
    # 进程只会互相抢同一批核, 反而变慢; 若真占到了别人的核, 是违规行为。
    # 所以这里做的是: **如实读出调度器分给我们多少核, 然后全部用满**。
    # 想多用核, 正确做法是在作业脚本里向调度器申请更多(见文件末尾的脚本模板)。
    #
    # 每种调度器都按多个环境变量依次回退 —— 原来的写法只看单一变量, 遇到
    # 常见的申请方式就会失效(例如 SLURM 用 --cpus-per-task 申请时
    # SLURM_NTASKS=1, 会导致只 spawn 0 个 worker)。
    # 手动覆盖: 设 JULIA_NWORKERS=N 可以强制指定 worker 数, 优先级最高。
    # ══════════════════════════════════════════════════════════════════════
    function _detect_cores()
        # 手动覆盖优先
        if haskey(ENV, "JULIA_NWORKERS")
            n = parse(Int, ENV["JULIA_NWORKERS"])
            return (n+1, "JULIA_NWORKERS 手动指定", "MANUAL")
        end
        # 依次尝试各调度器的多个变量, 取第一个 >1 的
        try_vars(vars) = begin
            for v in vars
                if haskey(ENV, v)
                    x = tryparse(Int, split(strip(ENV[v]), r"[,(x]")[1])
                    x !== nothing && x > 1 && return (x, v)
                end
            end
            return (0, "")
        end

        if haskey(ENV, "SGE_CELL") || haskey(ENV, "JOB_ID")
            n, v = try_vars(["NSLOTS", "SGE_HGR_processors", "OMP_NUM_THREADS"])
            n > 1 && return (n, "SGE: $(v)", "SGE")
            return (Sys.CPU_THREADS, "SGE 但读不到 NSLOTS, 退回本机核数", "SGE")

        elseif haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
            # 常见申请方式: -n(NTASKS) / -c(CPUS_PER_TASK) / --nodes+ntasks-per-node
            ntask = haskey(ENV,"SLURM_NTASKS")        ? something(tryparse(Int,ENV["SLURM_NTASKS"]),1)        : 1
            cpert = haskey(ENV,"SLURM_CPUS_PER_TASK") ? something(tryparse(Int,ENV["SLURM_CPUS_PER_TASK"]),1) : 1
            tot   = ntask * cpert
            tot > 1 && return (tot, "SLURM: NTASKS($(ntask)) × CPUS_PER_TASK($(cpert))", "SLURM")
            n, v = try_vars(["SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_NPROCS"])
            n > 1 && return (n, "SLURM: $(v)", "SLURM")
            return (Sys.CPU_THREADS, "SLURM 但读不到核数变量, 退回本机核数", "SLURM")

        elseif haskey(ENV, "PBS_JOBID")
            n, v = try_vars(["PBS_NP", "NCPUS", "PBS_NCPUS"])
            n > 1 && return (n, "PBS: $(v)", "PBS")
            if haskey(ENV, "PBS_NUM_PPN")
                ppn = something(tryparse(Int,ENV["PBS_NUM_PPN"]),1)
                nds = haskey(ENV,"PBS_NUM_NODES") ? something(tryparse(Int,ENV["PBS_NUM_NODES"]),1) : 1
                ppn*nds > 1 && return (ppn*nds, "PBS: PBS_NUM_PPN × PBS_NUM_NODES", "PBS")
            end
            return (Sys.CPU_THREADS, "PBS 但读不到核数变量, 退回本机核数", "PBS")

        elseif haskey(ENV, "LSB_JOBID")
            n, v = try_vars(["LSB_DJOB_NUMPROC", "LSB_MAX_NUM_PROCESSORS"])
            n > 1 && return (n, "LSF: $(v)", "LSF")
            return (Sys.CPU_THREADS, "LSF 但读不到核数变量, 退回本机核数", "LSF")
        end

        # 没有调度器: 登录节点或本机。
        # [FIX] 原来这里用 Sys.loadavg() 估"空闲核"然后几乎全占 —— 但
        # loadavg 在 Windows 上恒为 0, 会被判成"所有核都空闲", 直接 spawn
        # CPU线程数-1 个 worker, 实测把本机跑宕机了。真正的瓶颈也不是核数
        # 而是**内存**(每个 worker 要加载整个 Bloqade 栈, 约 1.5~2GB)。
        # 本地一律保守: 最多 4 个 worker, 再由下面的内存上限二次收紧。
        total = Sys.CPU_THREADS
        n = min(4, max(1, total - 1))
        return (n + 1, "无调度器(本机/登录节点): $(total) 核, 保守限制为 $(n) 个 worker", "LOCAL")
    end

    _ncores, _how, _sched = _detect_cores()
    _nworkers = max(1, _ncores - 1)   # 主进程占 1 个


    println("="^70)
    println("核数检测: $(_sched)")
    println("  依据: $(_how)")
    @printf("  调度器/系统给的核数: %d (主进程占 1, 最多可开 %d 个 worker)\n", _ncores, _ncores-1)
    # ── 内存上限 ────────────────────────────────────────────────────────
    # 每个 worker 要 using Bloqade/BloqadeODE/OrdinaryDiffEq 一整套, 常驻约
    # 1.5~2GB; 主进程还要再占一份。核数够不代表内存够 —— 实测本地按核数
    # spawn 直接把机器跑挂。这里按可用内存二次收紧, 宁可少几个 worker。
    # 想覆盖: 设 JULIA_WORKER_GB 调整"每 worker 预算"(默认 1.5 GB),
    # 或直接用 JULIA_NWORKERS 强制指定(会跳过所有自动限制)。
    # [注意] 用的是 free_memory 不是 total_memory —— 别的程序占着的内存
    # 我们抢不到。本机上浏览器/IDE 开着的话 free 会远小于 total, 这时被限到
    # 很少的 worker 是**正确**的, 强行覆盖会把机器跑挂(已经发生过一次)。
    if !haskey(ENV, "JULIA_NWORKERS")
        # 默认 1.5 GB/worker。这个数字**是估计值不是实测**, 所以下面 worker
        # 起来、包加载完之后会打印一次真实占用, 你可以照着调这个参数。
        gb_per_worker = something(tryparse(Float64, get(ENV,"JULIA_WORKER_GB","")), 1.5)
        free_gb  = Sys.free_memory()  / 2^30
        total_gb = Sys.total_memory() / 2^30
        # [FIX] 原公式是 (free*0.75 - 1份) / 1份 —— 先砍 25% 再扣主进程那份,
        # 两处保守叠加, 实际留了近一半内存不用。改成只做一次保守: 先如实扣掉
        # 主进程要占的那份, 剩余部分再乘一个安全系数(默认 0.85, 可用
        # JULIA_MEM_SAFETY 覆盖)。安全系数是必要的 —— free_memory 是瞬时值,
        # 别的程序随时可能再占走, 而且 BO 跑起来后每个 worker 还会因 GP 的
        # 核矩阵继续涨。
        # 有调度器时默认放宽到 1.0: 集群上内存由调度器(--mem / h_vmem)硬性
        # 分配, cgroups 会挡住超额, 而 Sys.free_memory() 读到的往往是**整个
        # 节点**的空闲内存(可能几百GB), 这个软性上限本来就不会生效, 留余量
        # 反而可能误伤。真正要保证的是作业脚本里申请够内存(每 worker ≥2GB)。
        # 本机无调度器时保持 0.85 —— 那里内存是真的紧张。
        default_safety = _sched == "LOCAL" ? 0.85 : 1.0
        safety = something(tryparse(Float64, get(ENV,"JULIA_MEM_SAFETY","")), default_safety)
        after_master = max(0.0, free_gb - gb_per_worker)     # 主进程那份
        usable_gb = after_master * safety
        mem_cap   = max(1, floor(Int, usable_gb / gb_per_worker))
        if mem_cap < _nworkers
            @printf("内存限制: 可用 %.1f/%.1f GB → 扣主进程 %.1f → 余 %.1f × 安全系数 %.2f = %.1f GB\n",
                    free_gb, total_gb, gb_per_worker, after_master, safety, usable_gb)
            @printf("            %.1f / %.1f(每worker) = 最多 %d 个 worker\n",
                    usable_gb, gb_per_worker, mem_cap)
            @printf("  worker 数从 %d 下调到 %d  (覆盖: JULIA_WORKER_GB / JULIA_MEM_SAFETY / JULIA_NWORKERS)\n",
                    _nworkers, mem_cap)
            _nworkers = mem_cap
        end
        # 可用内存占总内存比例很低时明确提醒: 这种情况多半是别的程序占着,
        # 关掉再跑能拿回不少 worker, 比调参数管用。
        if free_gb < total_gb*0.35
            @printf("  提示: 可用内存只有总量的 %.0f%%, 多半是浏览器/IDE 等占着。\n",
                    100*free_gb/total_gb)
            @printf("        关掉后重跑大约能开到 %d 个 worker。\n",
                    max(1, floor(Int, (total_gb*0.75 - gb_per_worker)/gb_per_worker)))
        end
    end
    @printf("  最终 worker 数: %d\n", _nworkers)
    haskey(ENV,"JOB_ID")        && println("  Job ID: $(ENV["JOB_ID"])")
    haskey(ENV,"SLURM_JOB_ID")  && println("  Job ID: $(ENV["SLURM_JOB_ID"])")
    haskey(ENV,"PBS_JOBID")     && println("  Job ID: $(ENV["PBS_JOBID"])")
    println("  (想改: 设环境变量 JULIA_NWORKERS=N 可强制指定 worker 数)")
    println("="^70)

    if _sched == "SLURM" && get(ENV,"JULIA_SLURM_MANAGER","") == "1"
        # 跨节点时需要 SlurmManager 才能把 worker 铺到其它节点上。
        # 默认不用它: 单节点情况下直接 addprocs 更省事、也不依赖 ClusterManagers。
        using ClusterManagers
        addprocs(SlurmManager(_nworkers))
    else
        # Bloqade 的 __init__ 会调 pyimport → CondaPkg.resolve() → 抢文件锁。
        # 让 worker 跳过 CondaPkg 解析: 主进程已经解析好了, worker 直接继承。
        get(ENV, "JULIA_CONDAPKG_BACKEND", "") == "" && (ENV["JULIA_CONDAPKG_BACKEND"] = "Null")
        # [FIX] 原来直接用 Base.active_project() 插进 exeflags, 出过 worker 上
        # "Package X not found" 的报错(报错栈里 Distributed/src/macros.jl 那几
        # 层, 说明是 worker 在跑 @everywhere 时失败, 不是主进程本身没装包)。
        # 常见原因: --project=<相对路径> 在某些启动方式/Julia版本下, 主进程
        # 自己解析对了, 但 Base.active_project() 返回的值传给 worker 后解析
        # 路径又不一样(尤其是从 notebook 里用 run(`julia ...`) 起子进程时,
        # 工作目录/环境变量继承容易跟直接在终端跑不一致)。
        # 这里先把路径转成绝对路径再打印出来自查, 万一还是不对, 至少能一眼
        # 看出问题出在哪, 不用再去啃 Distributed 内部的报错栈。
        proj_path = Base.active_project()
        proj_path === nothing && error("没有激活的 Julia 环境(Base.active_project() 是 nothing) —— " *
                                       "命令里要带 --project=ChenQAOA, 且要在包含 ChenQAOA/ 目录的路径下运行")
        proj_abs = abspath(proj_path)
        println("主进程环境: $(proj_abs)")
        addprocs(_nworkers; exeflags=`--project=$(proj_abs)`)

        # 起完 worker 立刻验一遍每个 worker 能不能读到同一个环境、能不能
        # import 关键包 —— 宁可现在就失败并给出明确原因, 也不要让它在几十行
        # 之后 @everywhere using ... 那里才报一个看不懂的远程调用栈。
        bad = String[]
        for w in workers()
            try
                wp = remotecall_fetch(() -> Base.active_project(), w)
                if wp === nothing || abspath(wp) != proj_abs
                    push!(bad, "worker $(w): 环境路径对不上 (worker看到的是 $(wp))")
                end
            catch e
                push!(bad, "worker $(w): 连环境路径都读不到 ($(sprint(showerror, e)))")
            end
        end
        if !isempty(bad)
            println("="^70)
            println("worker 环境校验失败:")
            for b in bad; println("  ", b); end
            println("最可能的原因: julia 找不到/不认得 --project=$(proj_abs) 这个绝对路径")
            println("(比如 juliaup 管理的多个 julia 版本之间环境隔离, 或者路径里有")
            println(" 权限/符号链接问题)。可以先试试手动跑:")
            println("  julia --project=$(proj_abs) -e 'using GenericTensorNetworks; println(\"ok\")'")
            println("="^70)
            error("worker 环境校验失败, 中止(避免几十分钟后才在 @everywhere 里报错)")
        end
    end

    println("Workers ready: $(nworkers())  IDs: $(workers())")
    println("="^70)
end

# ── 混合层 Rabi 幅度 Ωt(所有模式的演化函数都读 OMEGA_T_REF[]; 说明见 BOX_HI 处) ──
_omega_t = parse(Float64, get(ENV, "QAOA_OMEGA_T", "2.0"))
_omega_t > 0 || error("QAOA_OMEGA_T 必须为正数, 收到 $(_omega_t)")
@everywhere const QAOA_OMEGA_T = $_omega_t
@everywhere const OMEGA_T_REF = Ref(QAOA_OMEGA_T)
# GNN 预热的全局默认(所有模式共用, 包括 hybrid 基准 gnn_batch_qaoa_weighted):
#   QAOA_WS_MIXSIGN = −1 ⇒ Egger 精确版定制 mixer(失谐 −Ωt·q); +1(默认)= 原实现, 已有结果逐位不变
#   QAOA_WS_EPS = ε ⇒ GNN 概率夹到 [ε, 1−ε](Egger 正则化); 0(默认)= 不夹
_ws_mixsign = parse(Float64, get(ENV, "QAOA_WS_MIXSIGN", "1"))
_ws_mixsign in (1.0, -1.0) || error("QAOA_WS_MIXSIGN 只能是 1 或 -1")
_ws_eps_g = parse(Float64, get(ENV, "QAOA_WS_EPS", "0.0"))
(0.0 <= _ws_eps_g < 0.5) || error("QAOA_WS_EPS 必须在 [0, 0.5)")
@everywhere const WS_MIXSIGN_G = $_ws_mixsign
@everywhere const WS_EPS_G = $_ws_eps_g

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── Load packages on every worker ───────────────────────────────────────────
# Mirror the master-side imports. All packages needed inside compute_task()
# must appear here.
# JULIA_CONDAPKG_BACKEND=Null is set before addprocs(), so all workers
# inherit it and skip CondaPkg.resolve() when loading Bloqade.
# No lock race → safe to use @everywhere directly (loads into Main on workers).
# Note: @eval inside remotecall_fetch loads into an anonymous scope, not Main,
# causing UndefVarError for AtomList etc. @everywhere begin...end is correct.
@everywhere begin
    # worker 侧的 log_note: 只按级别决定要不要 println —— worker 不直接写日志
    # 文件(多个进程同时写一个文件会交错错乱), 输出经 stdout 流回主进程,
    # 由主进程统一落盘。级别通过环境变量传给 worker(addprocs 会继承环境)。
    const _W_RANK = Dict(:always=>3, :summary=>2, :detail=>1)
    _wlvl = get(ENV, "JULIA_CONSOLE_LEVEL", "summary")
    const W_CONSOLE_MIN = _wlvl == "always" ? :always : (_wlvl == "detail" ? :detail : :summary)
    # worker 侧缓冲: 每一行(不分级别)都存下来, 随任务返回值带回主进程统一落盘。
    # 低级别的 BO 细节因此既不占控制台配额, 又不会丢 —— worker 自己不写文件,
    # 因为多个进程同时写同一个文件会交错错乱。
    const W_LOG_BUF = String[]
    # [FIX] 这个函数原来也叫 log_note, 而 @everywhere 块在主进程上同样会执行,
    # 于是它把前面那个"会写日志文件"的主进程版 log_note 覆盖掉了 —— 结果主进程
    # 的日志一行都没进文件。改名区分: worker 用 wlog, 主进程用 log_note。
    function wlog(s::AbstractString, level::Symbol=:summary)
        push!(W_LOG_BUF, s)
        _W_RANK[level] >= _W_RANK[W_CONSOLE_MIN] && (println(s); flush(stdout))
        return nothing
    end

    using GenericTensorNetworks
    using Bloqade              # CondaPkg skipped via JULIA_CONDAPKG_BACKEND=Null
    using LinearAlgebra
    using Random
    using Statistics
    using Optim
    using SpecialFunctions
    using Printf
    using FFTW
    using Serialization
    using Dates
end
println("All workers loaded packages into Main ✓")

# ── 实测每个 worker 的真实内存占用 ────────────────────────────────────────
# 上面按 JULIA_WORKER_GB(默认1.5) 估算能开几个 worker, 但那是**估计值**。
# 这里在包全部加载完之后量一次真实 RSS, 让你能照着实测值调参, 而不是沿用
# 猜测。数字明显偏离默认值时会给出具体的调整建议。
# [FIX] 条件原来是 nworkers()>1, 结果只有 1 个 worker 时直接跳过 —— 恰恰是
# 内存最紧张、最需要看实测数据的场景反而没输出。改成 >=1。
if nworkers() >= 1
    try
        rss = [remotecall_fetch(() -> Sys.maxrss()/2^30, w) for w in workers()]
        mrss = Sys.maxrss()/2^30
        tot  = sum(rss) + mrss
        @printf("内存实测: 每 worker %.2f GB (最大 %.2f), 主进程 %.2f GB, 合计 %.2f GB / 本机 %.1f GB\n",
                sum(rss)/length(rss), maximum(rss), mrss, tot, Sys.total_memory()/2^30)
        est = something(tryparse(Float64, get(ENV,"JULIA_WORKER_GB","")), 1.5)
        avg = sum(rss)/length(rss)
        if avg > est*1.15
            @printf("  ⚠ 实测比预算(%.1f GB)高 %.0f%% —— 下次跑建议 JULIA_WORKER_GB=%.1f, 否则内存可能吃紧\n",
                    est, 100*(avg/est-1), ceil(avg*10)/10)
        elseif avg < est*0.7
            @printf("  实测比预算(%.1f GB)低不少 —— 可以设 JULIA_WORKER_GB=%.1f 开更多 worker\n",
                    est, ceil(avg*10)/10)
        end
        println("  (注意: 这是包加载完的基线; 跑起来后每个 worker 还会因 BO 的 GP 矩阵再涨一些)")
    catch e
        @warn "内存实测失败(不影响计算)" exception=e
    end
end

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── ALGORITHM DEFINITIONS (broadcast to all workers via @everywhere) ─────────

@everywhere begin

# ── GLOBAL CONSTANTS ────────────────────────────────────────────────────────
# BO 的三个"会改变数值结果"的配置项集中定义在这里, 作为单一事实来源 ——
# checkpoint 的目录名会带上它们的指纹, 改动后旧结果自动失效, 不会静默串数据。
const BO_DIM_THRESH = 6      # dim >= 这个值时启用高维档配置(候选池、精修尺度等)
const BO_BUDGET_A   = 100    # budget = A*dim + B*dim
const BO_BUDGET_B   = 5
const C6_VAL  = 2π * 862690
const P_VAL   = 3
const DIM     = 2 * P_VAL
const N_EDGES = 12         # Q3 hypercube: 12 edges, max-cut = 12
const LOWER   = zeros(DIM)
const UPPER   = [3π, π, 3π, π, 3π, π]         # γ ∈ [0,3π]: 1.5 periods (prev best γ₃=2π hit boundary); β ∈ [0,π]

# Joint (d, γ, β) parameter space for 7D BO
# First dimension is atomic spacing d in μm; rest are QAOA angles.
const DIM_JOINT  = DIM + 1
const LOWER_JOINT = vcat([9.0], zeros(DIM))           # d ∈ [9, 15] μm
const UPPER_JOINT = vcat([18.0], [3π, π, 3π, π, 3π, π])         # d ∈ [9,18] μm (10/20 seeds hit d=15 boundary); γ ∈ [0,3π]; β ∈ [0,π]
const D_CANDIDATES_W = [12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
const SEEDS_W        = collect(1:20)   # must match master SEEDS

function task_params_w(task_id::Int)
    n_s   = length(SEEDS_W)
    d_idx = (task_id - 1) ÷ n_s + 1
    s_idx = (task_id - 1) % n_s + 1
    return D_CANDIDATES_W[d_idx], SEEDS_W[s_idx]
end

# ── HALTON LOW-DISCREPANCY SEQUENCE ─────────────────────────────────────────
const HALTON_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # 10 primes: covers p=1..5 (DIM=2..10) and joint 7D

function vdc(n::Int, b::Int)
    result = 0.0; denom = 1.0
    while n > 0
        denom  *= b
        result += (n % b) / denom
        n       = n ÷ b
    end
    return result
end

function halton_samples(n::Int, lo::Vector, hi::Vector; skip::Int=20)
    d = length(lo)
    pts = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        pt = [lo[dim] + vdc(i+skip, HALTON_BASES[dim]) * (hi[dim]-lo[dim])
              for dim in 1:d]
        pts[i] = pt
    end
    return pts
end

# ── PHYSICS LAYER — Q3 hypercube atoms & edges ──────────────────────────────
function make_cube_atoms(d::Float64)
    positions = Tuple{Float64,Float64,Float64}[]
    for i in 0:7
        x = Float64((i     ) & 1) * d
        y = Float64((i >> 1) & 1) * d
        z = Float64((i >> 2) & 1) * d
        push!(positions, (x, y, z))
    end
    return AtomList(positions)
end

function build_edges_by_radius(atoms::AtomList, radius::Real)
    edges = Tuple{Int,Int}[]
    pos   = [(x,y,z) for (x,y,z) in atoms]
    for i in 1:length(atoms), j in (i+1):length(atoms)
        dx=pos[i][1]-pos[j][1]; dy=pos[i][2]-pos[j][2]; dz=pos[i][3]-pos[j][3]
        sqrt(dx^2+dy^2+dz^2) <= radius && push!(edges,(i,j))
    end
    return edges
end

# ── QUANTUM SIMULATION ───────────────────────────────────────────────────────
function maxcut_ev(reg::ArrayReg, edges)
    sv = reg.state; val = 0.0
    for b in 0:(2^nqubits(reg)-1)
        prob = abs2(sv[b+1]); prob > 1e-12 || continue
        cut  = sum((b>>(u-1))&1 != (b>>(v-1))&1 ? 1 : 0 for (u,v) in edges)
        val += prob * cut
    end
    return val
end

function loss_fn(x::AbstractVector{T}, atoms, edges, n_atoms) where T
    @assert length(x) % 2 == 0
    p_cur = length(x) ÷ 2
    Ω0=5π; t0=π/(2*Ω0); Ωt=OMEGA_T_REF[]; C6=2π*862690
    pos = [atoms[i] for i in 1:n_atoms]
    # [FIX] 原来只对声明的边求和(sw只加edges里的ew), 论文(Bauer et al. 2024,
    # Eq.10前)明确要求 Δᵢ=2Σ_{j≠i}wᵢⱼ 对*所有*其他原子求和, 这样才能精确抵消
    # 线性Z项 (物理上任意两个原子之间都有范德华相互作用, 不只是声明的边)。
    sw = zeros(n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        xi,yi,zi=pos[i]; xj,yj,zj=pos[j]
        r2=(xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        sw[i] += C6/(4*r2^3)
    end
    Δt  = 2.0*sw
    dur = vcat(t0, abs.(x))
    clk = [0.0, cumsum(dur)...]
    Ωv  = vcat(Ω0, [repeat([0.0, Ωt], p_cur)...])
    # FIX: zeros(T, 2*p_cur) replaces parametric T[0.0,0.0] literal
    ϕv  = vcat(-0.5π, zeros(T, 2 * p_cur))
    td  = clk[end]
    Δwf = [piecewise_constant(clocks=[0.0,td], values=[Δt[i]]) for i in 1:n_atoms]
    h   = rydberg_h(atoms; Δ=Δwf,
                    Ω=piecewise_constant(clocks=clk, values=Ωv),
                    ϕ=piecewise_constant(clocks=clk, values=ϕv))
    reg  = zero_state(Complex{T}, n_atoms)
    prob = KrylovEvolution(reg, clk, h); emulate!(prob)
    return -maxcut_ev(prob.reg, edges)
end

"""loss_fn 的带权版本 —— 跟原来的 loss_fn 逐行一致 (硬件原生的探测距离偏置
   ew/sw/Δt 完全不变, 仍然只由原子物理距离决定, 跟逻辑权重 weights 无关,
   这是刻意保留的设计: 探测偏置代表"硬件原生的纠缠资源", 逻辑权重只体现在
   最终测量/优化的目标函数上)。唯一改动: 最后一行从无权 maxcut_ev 换成
   带权 maxcut_ev_weighted。"""
function loss_fn_weighted(x::AbstractVector{T}, atoms, edges, weights::Vector{Float64},
                           n_atoms) where T
    @assert length(x) % 2 == 0
    p_cur = length(x) ÷ 2
    Ω0=5π; t0=π/(2*Ω0); Ωt=OMEGA_T_REF[]; C6=2π*862690
    pos = [atoms[i] for i in 1:n_atoms]
    # [FIX] 同 loss_fn: 对所有原子对求和, 不只是声明的边
    sw = zeros(n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        xi,yi,zi=pos[i]; xj,yj,zj=pos[j]
        r2=(xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        sw[i] += C6/(4*r2^3)
    end
    Δt  = 2.0*sw
    dur = vcat(t0, abs.(x))
    clk = [0.0, cumsum(dur)...]
    Ωv  = vcat(Ω0, [repeat([0.0, Ωt], p_cur)...])
    ϕv  = vcat(-0.5π, zeros(T, 2 * p_cur))
    td  = clk[end]
    Δwf = [piecewise_constant(clocks=[0.0,td], values=[Δt[i]]) for i in 1:n_atoms]
    h   = rydberg_h(atoms; Δ=Δwf,
                    Ω=piecewise_constant(clocks=clk, values=Ωv),
                    ϕ=piecewise_constant(clocks=clk, values=ϕv))
    reg  = zero_state(Complex{T}, n_atoms)
    prob = KrylovEvolution(reg, clk, h); emulate!(prob)
    return -maxcut_ev_weighted(prob.reg, edges, weights)
end

# ── GSM+CSM SPECTRAL MIXTURE KERNEL ─────────────────────────────────────────
function safe_chol(M::Matrix{Float64}, j0::Float64)
    j = j0
    for _ in 1:8
        try; return cholesky(Symmetric(M+j*I)).L, j; catch; j*=10.0; end
    end
    return cholesky(Symmetric(j*I(size(M,1)))).L, j
end

function spectral_mixture_kernel(X1::Matrix{Float64}, X2::Matrix{Float64},
                                  θ::Vector{Float64}, Q_g::Int, Q_c::Int, D::Int)
    n1, n2 = size(X1,2), size(X2,2)
    K = zeros(n1, n2); idx = 1
    # FIX: Δd must be (n1, n2) to match exp_acc/cos_acc/K.
    # X1[d:d,:]  is (1,n1); X2[d:d,:] is (1,n2).
    # X1[d:d,:]' is (n1,1); X2[d:d,:] is (1,n2).
    # (n1,1) .- (1,n2) broadcasts to (n1,n2) — correct shape.
    # The old code used X1[d:d,:] .- X2[d:d,:]' = (1,n1)-(n2,1) = (n2,n1)
    # which caused DimensionMismatch whenever n1 ≠ n2.
    for _ in 1:Q_g
        w = exp(θ[idx]); idx += 1
        ℓ = exp.(θ[idx:idx+D-1]); idx += D
        μ = θ[idx:idx+D-1];       idx += D
        exp_acc = zeros(n1,n2); cos_acc = zeros(n1,n2)
        for d in 1:D
            Δd = X1[d:d,:]' .- X2[d:d,:]   # (n1,1) .- (1,n2) → (n1,n2)
            exp_acc .+= (Δd./ℓ[d]).^2; cos_acc .+= μ[d].*Δd
        end
        K .+= w .* exp.(-2π^2 .* exp_acc) .* cos.(2π .* cos_acc)
    end
    for _ in 1:Q_c
        w  = exp(θ[idx]); idx += 1
        γ  = exp.(θ[idx:idx+D-1]); idx += D
        x0 = θ[idx:idx+D-1];       idx += D
        exp_acc = zeros(n1,n2); cos_acc = zeros(n1,n2)
        for d in 1:D
            Δd = X1[d:d,:]' .- X2[d:d,:]   # (n1,1) .- (1,n2) → (n1,n2)
            exp_acc .+= abs.(Δd).*γ[d]; cos_acc .+= x0[d].*Δd
        end
        K .+= w .* exp.(-2π .* exp_acc) .* cos.(2π .* cos_acc)
    end
    return K
end

# ══════════════════════════════════════════════════════════════════════════
# ── [GP v2] 版本开关 ────────────────────────────────────────────────────────
# QAOA_GP_VERSION=1 (默认): 与旧版逐比特一致 —— 所有已有 checkpoint / 基准照常有效。
# QAOA_GP_VERSION=2: 三处修正 + 超参数拟合加强, 只改变 GP 内部, 调用方式不变:
#   (1) y 标准化: 拟合前 y ← (y-均值)/标准差。原来直接用 -⟨C⟩≈-7…-10 去拟合
#       零均值 GP, 要浪费一个混合成分去解释"整体在 -8 附近"。gp_pred 返回时
#       **还原回原始单位**, 所以 EI 的 ξ、当前最优 by 等仍按原始单位理解, 语义不变。
#   (2) 输入归一化(调用方传 xscale=hi-lo 时): x ← x/量程, 两个方向都变成 [0,1]。
#       [关键注意] 频率参数 μ(GSM)/x0(CSM) 的单位是"周期/单位长度", 归一化后同一个
#       物理频率的数值会放大 量程 倍(γ 方向 3π≈9.4 倍)。所以频率的上下界和先验
#       宽度同步乘以量程 —— 否则高频结构(能量曲面周期约 0.5)会被先验压掉。
#   (4) Thompson 采样改为**联合**后验采样: 原来每个候选点独立按边缘方差抽签,
#       等于偏爱"没去过的地方"; 现在从联合后验抽整幅函数, 取其最低点。候选点多于
#       TS_JOINT_MAX 时随机取子集, 控制 O(m³) 开销。
#   (3) 超参数拟合: NelderMead 迭代 400→GP_V2_HP_ITERS, 重启次数 +GP_V2_EXTRA_RESTARTS。
# 任何一项改动都会改变 BO 轨迹 → 批量模式下 checkpoint 目录名加 GP_CFG_TAG 后缀。
# worker 通过继承环境变量拿到同样的版本号(addprocs 会继承主进程环境)。
const GP_VERSION           = parse(Int, get(ENV, "QAOA_GP_VERSION", "1"))
const GP_V2_HP_ITERS       = 800
const GP_V2_EXTRA_RESTARTS = 1
const TS_JOINT_MAX         = 1000
const GP_CFG_TAG           = "gp2i$(GP_V2_HP_ITERS)r$(GP_V2_EXTRA_RESTARTS)"

_gp_xs(X, xscale) = xscale === nothing ? X : X ./ xscale      # D×n 按行除以量程
function _gp_ystd(y)
    ym = sum(y) / length(y)
    sd = length(y) > 1 ? std(y) : 1.0
    (isfinite(sd) && sd > 1e-12) || (sd = 1.0)                 # 全部相同的 y: 不缩放
    return ym, sd
end
"""超参数箱约束与先验宽度。xscale=nothing 时与旧版完全相同。"""
function _spectral_bounds(Q_g::Int, Q_c::Int, D::Int, xscale)
    n_p = (Q_g+Q_c)*(1+2D)
    lo = -6.0*ones(n_p); hi = 6.0*ones(n_p); ps = ones(n_p)
    tidx = 1
    # 参数排布: 每个成分 [log w, D 个 log 尺度, D 个频率]; GSM 在前, CSM 在后
    for _ in 1:(Q_g+Q_c)
        tidx += 1+D
        for d in 1:D
            sc = xscale === nothing ? 1.0 : xscale[d]
            lo[tidx] = -2π*sc; hi[tidx] = 2π*sc; ps[tidx] = sc; tidx += 1
        end
    end
    return lo, hi, ps
end

"""GSM+CSM 核超参数的 MAP 拟合。
   [新增] θ_init: 热启动。一轮 BO 内要重拟合几十次(map_every=18), 原来**每次
   都从完全随机的 θ₀ 开始**, 把上一次辛苦拟合出来的结果整个丢掉 —— 而相邻两次
   拟合之间只多了十几个观测点, 最优超参数几乎不会跳变。传入上次的 θ 作为第一个
   起点(其余重启仍随机, 保留跳出局部解的能力), 同样的迭代预算下能收敛得更好,
   也更省时间。默认 nothing ⇒ 行为与改动前**完全一致**, 不影响已有的联合BO基线。
   [GP v2] gpv / xscale 见上方版本开关说明; 必须与 build_spectral_gp 传同样的值。"""
function map_spectral_hp(X::Matrix{Float64}, y::Vector{Float64},
                          Q_g::Int, Q_c::Int, D::Int; nr::Int=2,
                          θ_init::Union{Nothing,Vector{Float64}}=nothing,
                          gpv::Int=GP_VERSION,
                          xscale::Union{Nothing,Vector{Float64}}=nothing,
                          hp_iters::Int=0, hp_restarts::Int=0, hp_rerun::Bool=false)
    # 以下三个参数只给核消融用, 默认值 = 行为完全不变:
    #   hp_iters>0    覆盖 NelderMead 迭代上限(实测: 上限不是瓶颈, NM 会按自身收敛判据提前停)
    #   hp_restarts>0 覆盖总重启次数(这才是真正能增加拟合力度的旋钮)
    #   hp_rerun      每次 NM 停下后, 从停下的点重新展开单纯形再跑一次 —— 高维 NM
    #                 常因单纯形塌缩而"假收敛", 重启单纯形是标准补救办法
    v2  = gpv >= 2
    xs_ = v2 ? xscale : nothing
    Xn  = _gp_xs(X, xs_)
    yz  = if v2
        ym, sd = _gp_ystd(y); (y .- ym) ./ sd
    else
        y
    end
    lo, hi, ps = _spectral_bounds(Q_g, Q_c, D, xs_)
    n_p   = length(lo)
    iters = hp_iters > 0 ? hp_iters : (v2 ? GP_V2_HP_ITERS : 400)
    nr_ef = hp_restarts > 0 ? hp_restarts : (v2 ? nr + GP_V2_EXTRA_RESTARTS : nr)
    bθ, bv = nothing, Inf
    for ri in 1:nr_ef
        # 第一个起点用热启动的 θ(如果给了且维度对得上), 其余保持随机重启
        θ0 = if ri == 1 && θ_init !== nothing && length(θ_init) == n_p
            # [FIX] 不能只 clamp 到 [lo,hi] —— Fminbox 要求初始点在箱体**内部**,
            # 落在边界上会刷 "Initial position cannot be on the boundary of the box"
            # 警告(它自己会往内挪, 结果正确, 但日志被刷屏, 每个 worker 都刷)。
            # 上次拟合出来的 θ 在箱约束下经常正好停在边界, 所以这里主动往内缩
            # 一个小量。1e-6 相对于 [-6,6] / [-2π,2π] 的箱体宽度可以忽略。
            clamp.(θ_init, lo .+ 1e-6, hi .- 1e-6)
        else
            lo .+ rand(n_p) .* (hi.-lo)
        end
        function nll(θ)
            try
                Km = spectral_mixture_kernel(Xn,Xn,θ,Q_g,Q_c,D)
                L,_ = safe_chol(Km,1e-4)
                α = L'\(L\yz); ld = 2sum(log.(diag(L))); n = length(yz)
                ll = -0.5*dot(yz,α)-0.5*ld-0.5*n*log(2π)
                # 先验: 各参数 N(0,1); v2 归一化输入后频率参数的先验宽度 = 量程(物理频率先验不变)
                pen = v2 ? 0.5*sum((θ ./ ps).^2) : 0.5*sum(θ.^2)
                v  = -(ll-pen)
                return isfinite(v) ? v : 1e10   # never return Inf/NaN
            catch
                return 1e10   # finite fallback — gradient stays well-defined
            end
        end
        # Use NelderMead (derivative-free) inside Fminbox.
        # LBFGS computes gradients via finite differences: (nll(θ+h)-nll(θ))/h.
        # When nll returns Inf, the gradient becomes Inf or NaN, causing
        # HagerZhang linesearch AssertionError: isfinite(phi_c) && isfinite(dphi_c).
        # NelderMead never requests gradients → immune to non-finite nll values.
        r = optimize(nll, lo, hi, θ0, Fminbox(NelderMead()),
                     Optim.Options(iterations=iters, show_trace=false))
        if hp_rerun
            θr = clamp.(Optim.minimizer(r), lo .+ 1e-6, hi .- 1e-6)
            r2 = optimize(nll, lo, hi, θr, Fminbox(NelderMead()),
                          Optim.Options(iterations=iters, show_trace=false))
            Optim.minimum(r2) < Optim.minimum(r) && (r = r2)
        end
        Optim.minimum(r)<bv && (bv=Optim.minimum(r); bθ=Optim.minimizer(r))
    end
    return isnothing(bθ) ? lo.+0.5.*(hi.-lo) : bθ
end

function _prior_var(θ::Vector{Float64}, Q_g::Int, Q_c::Int, D::Int)
    v=0.0; tidx=1
    for _ in 1:(Q_g+Q_c); v+=exp(θ[tidx]); tidx+=1+2D; end
    return v
end

struct SpectralGP
    L::LowerTriangular{Float64,Matrix{Float64}}
    α::Vector{Float64}; X::Matrix{Float64}; θ::Vector{Float64}
    Q_g::Int; Q_c::Int; D::Int; sc::Float64
    ym::Float64; ysd::Float64                       # [v2] y 标准化参数(v1: 0,1 且不使用)
    xs::Union{Nothing,Vector{Float64}}              # [v2] 输入量程(X 已按它归一化)
    v::Int                                          # GP 版本
end
function build_spectral_gp(X,y,θ,Q_g,Q_c,D; gpv::Int=GP_VERSION,
                           xscale::Union{Nothing,Vector{Float64}}=nothing)
    v2 = gpv >= 2; xs_ = v2 ? xscale : nothing
    Xn = _gp_xs(X, xs_)
    ym, sd = v2 ? _gp_ystd(y) : (0.0, 1.0)
    yz = v2 ? (y .- ym) ./ sd : y
    Km=spectral_mixture_kernel(Xn,Xn,θ,Q_g,Q_c,D); L,_=safe_chol(Km,1e-4)
    SpectralGP(L,L'\(L\yz),Xn,θ,Q_g,Q_c,D,_prior_var(θ,Q_g,Q_c,D), ym, sd, xs_, gpv)
end
"""后验均值与标准差, **始终以原始单位返回**(v2 内部标准化, 这里还原)。"""
function gp_pred(gp::SpectralGP, Xn::Matrix{Float64})
    Xq=_gp_xs(Xn, gp.xs)
    Kx=spectral_mixture_kernel(gp.X,Xq,gp.θ,gp.Q_g,gp.Q_c,gp.D)
    v=gp.L\Kx
    μ = Kx'*gp.α
    σ = sqrt.(max.(gp.sc.-vec(sum(v.^2;dims=1)),0.0))
    return gp.v >= 2 ? (gp.ym .+ gp.ysd .* μ, gp.ysd .* σ) : (μ, σ)
end
function ts_acq(gp::SpectralGP, C::Matrix{Float64}; ns=5)
    Cq=_gp_xs(C, gp.xs)
    Kx=spectral_mixture_kernel(gp.X,Cq,gp.θ,gp.Q_g,gp.Q_c,gp.D)
    v=gp.L\Kx; μ=Kx'*gp.α
    if gp.v >= 2
        # [v2] 联合后验采样: 抽 ns 幅完整的函数样本, 取所有样本里的最低点
        m   = size(C,2)
        sub = m > TS_JOINT_MAX ? randperm(m)[1:TS_JOINT_MAX] : collect(1:m)
        Kcc = spectral_mixture_kernel(Cq[:,sub],Cq[:,sub],gp.θ,gp.Q_g,gp.Q_c,gp.D)
        Σ   = Kcc .- v[:,sub]'*v[:,sub]; Σ = (Σ .+ Σ') ./ 2
        Lc,_ = safe_chol(Σ, 1e-8*max(gp.sc, 1e-12))
        bi=0; bs=Inf
        for _ in 1:ns
            smp = μ[sub] .+ Lc*randn(length(sub)); k = argmin(smp)
            smp[k] < bs && (bs = smp[k]; bi = sub[k])
        end
        return bi
    end
    Σd=max.(gp.sc.-vec(sum(v.^2;dims=1)),1e-10)
    bi=0; bs=Inf
    for _ in 1:ns
        s=μ.+sqrt.(Σd).*randn(size(C,2)); idx=argmin(s)
        s[idx]<bs && (bs=s[idx]; bi=idx)
    end
    return bi
end

Φ(z) = 0.5*SpecialFunctions.erfc(-z/sqrt(2))
φ(z) = exp(-0.5*z^2)/sqrt(2π)
ei_acq(μ,σ,yb;ξ=0.01) = begin
    imp=yb.-μ.-ξ; Z=imp./(σ.+1e-9)
    return max.(imp.*Φ.(Z).+σ.*φ.(Z),0.0)
end
lcb_acq(μ,σ;β=0.5) = μ.-β.*σ

# ── LAYER 2: meta-initialisation ────────────────────────────────────────────
mutable struct InitLayer
    history_d::Vector{Float64}; history_x::Vector{Vector{Float64}}
end
InitLayer() = InitLayer(Float64[],Vector{Float64}[])

function predict_x0(layer::InitLayer, d::Float64; T_scale=3.0)
    isempty(layer.history_d) && return nothing
    dists=abs.(layer.history_d.-d); weights=exp.(-dists./T_scale)
    weights./=sum(weights)
    return clamp.(sum(w*x for (w,x) in zip(weights,layer.history_x)),LOWER,UPPER)
end
function update_init_layer!(layer::InitLayer, d::Float64, best_x::Vector{Float64})
    push!(layer.history_d,d); push!(layer.history_x,copy(best_x))
end

# ── LAYER 1: spacing GP-EI ───────────────────────────────────────────────────
function spacing_rbf_K(d1::Vector,d2::Vector;l=3.0,σ2=1.0)
    M=zeros(length(d1),length(d2))
    for i in 1:length(d1), j in 1:length(d2)
        M[i,j]=σ2*exp(-0.5*(d1[i]-d2[j])^2/l^2)
    end
    return M
end
function spacing_gp_predict(D_obs::Vector,y_obs::Vector,d_cands::Vector;noise=0.10)
    K=spacing_rbf_K(D_obs,D_obs)+noise*I
    Ks=spacing_rbf_K(D_obs,d_cands); Kss=spacing_rbf_K(d_cands,d_cands)
    L=cholesky(Symmetric(K)).L; α=L'\(L\y_obs)
    μ=Ks'*α; v=L\Ks
    return μ, sqrt.(max.(diag(Kss).-vec(sum(v.^2;dims=1)),0.0))
end
Φs(z)=0.5*erfc(-z/sqrt(2)); φs(z)=exp(-0.5*z^2)/sqrt(2π)
function ei_spacing(μ,σ,y_best;ξ=0.01)
    imp=y_best.-μ.-ξ; Z=imp./(σ.+1e-9)
    return max.(imp.*Φs.(Z).+σ.*φs.(Z),0.0)
end

end # @everywhere algorithm definitions

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── PER-WORKER TASK FUNCTION ────────────────────────────────────────────────
# Follows the template pattern exactly:
#   @everywhere function compute_task(params) ... return (...) end
# params is a NamedTuple so workers need no shared state.

@everywhere function compute_task(params)
    task_id   = params.task_id
    d, seed   = task_params_w(task_id)
    budget    = params.budget
    n_init    = params.n_init

    lo = copy(LOWER); hi = copy(UPPER); dim = DIM
    pct = round(task_id / (length(D_CANDIDATES_W)*length(SEEDS_W)) * 100, digits=1)
    println("Starting task $(lpad(task_id,2)) [$(pct)%]  d=$(d) μm  seed=$(seed)  — $(Dates.format(now(),"HH:MM:SS"))")
    flush(stdout)

    atoms   = make_cube_atoms(d)
    edges   = build_edges_by_radius(atoms, d*1.2)
    n_atoms = length(atoms); n_edges = length(edges)

    cache=Dict{Vector{Float64},Float64}(); eval_count=Ref(0)
    function ev!(x)
        xs=clamp.(x,lo,hi); key=round.(xs,digits=8)
        haskey(cache,key) && return cache[key]
        eval_count[]+=1
        try
            lv=loss_fn(xs,atoms,edges,n_atoms)
            if isfinite(lv); cache[key]=lv; return lv; end
        catch e; println("  eval error: $e"); end
        return 1000.0
    end

    # Halton warm-start
    n_h=n_init; halton_pts=halton_samples(n_h,lo,hi;skip=seed%50)
    init_xs=[clamp.(pt.+randn(dim).*0.10.*(hi.-lo),lo,hi) for pt in halton_pts]
    init_ys=Float64[ev!(x) for x in init_xs]
    X=hcat(init_xs...); y=copy(init_ys)
    by=minimum(init_ys); bx=init_xs[argmin(init_ys)]

    # n_bo_iter capped to budget-n_init: BO can only add (budget-n_init)
    # new evaluations before hitting budget. Running more iterations only
    # wastes time on GP kernel computations with cached (no-op) ev! calls.
    n_bo_iter = budget - n_init; total_budget=budget
    curve=fill(by,total_budget)
    for i in 1:n_init; curve[i]=minimum(init_ys[1:i]); end
    println("  warm-start done  ar=$(round(-by/n_edges,digits=4))")

    # Early exit: if warm-start ar < 0.35, this d-value is too weak
    # (U_ij << 1, very slow evaluations, BO unlikely to help much).
    # Return warm-start result immediately to save cluster time.
    if -by/n_edges < 0.35
        println("  [early exit] ar=$(round(-by/n_edges,digits=4)) < 0.35 — skipping BO")
        for i in 2:total_budget; curve[i]=min(curve[i],curve[i-1]); end
        ar_early = -by/n_edges
        elapsed_early = round(time()-params.t_start, digits=1)
        return (task_id, d, seed, by, bx, curve, eval_count[], n_edges, ar_early, elapsed_early)
    end

    Q_g=2; Q_c=2; map_every=20; n_cand=300
    phase1_end=0.25; phase2_end=0.65; stag_limit=30; refine_every=20
    θ=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=2); stag=0

    for iter in 1:n_bo_iter
        t=iter/n_bo_iter; eval_idx=min(n_init+iter,total_budget)
        iter>1 && iter%map_every==0 && (θ=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=2))
        gp=build_spectral_gp(X,y,θ,Q_g,Q_c,dim)
        C=lo.+rand(dim,n_cand).*(hi.-lo)
        if t<phase1_end
            bi=ts_acq(gp,C;ns=10); xn=C[:,bi]; lab="TS"
        elseif t<phase2_end
            ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]; lab="EI"
        else
            β=max(exp(-2.0*(t-phase2_end)/(1-phase2_end)),0.1)
            μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]; lab="LCB"
        end
        yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
        if yn<by
            by=yn; bx=copy(xn); stag=0
            println("    eval=$(n_init+iter) [$lab] ✓ ar=$(round(-by/n_edges,digits=4))")
            flush(stdout)
        else; t>=phase1_end && (stag+=1); end
        curve[eval_idx]=by
        if stag>=stag_limit
            for _ in 1:2
                xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                X=hcat(X,xr); push!(y,yr); yr<by && (by=yr; bx=copy(xr))
            end
            stag=0; curve[eval_idx]=by
        end
        if iter%refine_every==0 && t>=phase1_end
            for sc in [0.05,0.20,0.50,0.80,1.0]
                xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                nm_iters = dim >= 8 ? 100 : 50   # deeper refinement in high-D
                r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                           Optim.Options(iterations=nm_iters,f_reltol=1e-6,show_trace=false))
                yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                if yr<by-1e-6
                    println("    [MRLR sc=$sc] ✓ $(round(-by/n_edges,digits=4)) → $(round(-yr/n_edges,digits=4))")
                    flush(stdout)
                    by=yr; bx=xr; stag=0; X=hcat(X,xr); push!(y,yr)
                end
            end
            curve[eval_idx]=by
        end
    end
    for i in 2:total_budget; curve[i]=min(curve[i],curve[i-1]); end

    elapsed = round(time()-params.t_start, digits=1)
    ar = -by/n_edges
    println("Finished task $(lpad(task_id,2))  d=$(d) μm  seed=$(seed)  ar=$(round(ar,digits=4))  evals=$(eval_count[])  time=$(elapsed)s — $(Dates.format(now(),"HH:MM:SS"))")
    flush(stdout)

    # Return tuple — same consistent type for every task
    return (task_id, d, seed, by, bx, curve, eval_count[], n_edges, ar, elapsed)
end

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── PHASE 2 WORKER FUNCTION ─────────────────────────────────────────────────
# Dedicated function for Phase 2 that accepts explicit d and seed parameters
# instead of looking them up from a task_id table.
# This fixes the critical bug where Phase 2 ignored d_next values.

@everywhere function compute_phase2_task(d::Float64, seed::Int,
                                          budget::Int, n_init::Int,
                                          x0_hint::Union{Vector{Float64},Nothing},
                                          t_start::Float64)
    lo = copy(LOWER); hi = copy(UPPER); dim = DIM

    println("  [Phase 2] d=$(round(d,digits=2)) μm  seed=$(seed)  — $(Dates.format(now(),"HH:MM:SS"))")
    flush(stdout)

    atoms   = make_cube_atoms(d)
    edges   = build_edges_by_radius(atoms, d * 1.2)
    n_atoms = length(atoms); n_edges = length(edges)

    Random.seed!(seed)
    cache = Dict{Vector{Float64},Float64}(); eval_count = Ref(0)
    function ev!(x)
        xs  = clamp.(x, lo, hi); key = round.(xs, digits=8)
        haskey(cache, key) && return cache[key]
        eval_count[] += 1
        try
            lv = loss_fn(xs, atoms, edges, n_atoms)
            if isfinite(lv); cache[key] = lv; return lv; end
        catch e; println("  eval error: $e"); end
        return 1000.0
    end

    # Halton warm-start, optionally prepend x0_hint from Layer 2
    n_h = x0_hint !== nothing ? n_init - 1 : n_init
    halton_pts = halton_samples(n_h, lo, hi; skip = seed % 50)
    init_xs = [clamp.(pt .+ randn(dim) .* 0.10 .* (hi .- lo), lo, hi)
               for pt in halton_pts]
    x0_hint !== nothing && push!(init_xs, clamp.(x0_hint, lo, hi))

    init_ys = Float64[ev!(x) for x in init_xs]
    X = hcat(init_xs...); y = copy(init_ys)
    by = minimum(init_ys); bx = init_xs[argmin(init_ys)]

    total_budget = budget
    n_bo_iter = budget - n_init   # cap to avoid wasted GP iterations
    curve = fill(by, total_budget)
    for i in 1:n_init; curve[i] = minimum(init_ys[1:i]); end
    println("    warm-start done  ar=$(round(-by/n_edges, digits=4))")

    # BO strategy — all values proven by Q3_local_d14.jl:
    #   p=4 mean=0.877 best=0.893, p=5 mean=0.929 best=0.950  (exceeds GW)
    Q_g=2; Q_c=2
    n_cand       = dim >= 8 ? 800  : 400          # candidates per BO iter
    map_every    = dim >= 8 ? 12   : 18            # HP refit frequency
    phase1_end   = dim >= 8 ? 0.45 : 0.30         # TS exploration fraction
    phase2_end   = dim >= 8 ? 0.78 : 0.68         # EI exploitation fraction
    nr_hp        = dim >= 8 ? 4    : 2             # HP random restarts
    stag_limit   = dim >= 8 ? 25   : 20            # stagnation patience
    refine_every = 15                               # MRLR trigger frequency
    nm_iters     = dim >= 8 ? 200  : 100           # NelderMead depth
    refine_scales = dim >= 8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
    θ = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=nr_hp); stag = 0

    for iter in 1:n_bo_iter
        t = iter / n_bo_iter; eval_idx = min(n_init + iter, total_budget)
        iter > 1 && iter % map_every == 0 &&
            (θ = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=2))
        gp = build_spectral_gp(X, y, θ, Q_g, Q_c, dim)
        C  = lo .+ rand(dim, n_cand) .* (hi .- lo)
        if t < phase1_end
            bi = ts_acq(gp, C; ns=10); xn = C[:,bi]; lab = "TS"
        elseif t < phase2_end
            ξ   = 0.05 * exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ = gp_pred(gp, C); xn = C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]; lab = "EI"
        else
            β   = max(exp(-2.0*(t-phase2_end)/(1-phase2_end)), 0.1)
            μ,σ = gp_pred(gp, C); xn = C[:,argmin(lcb_acq(μ,σ;β=β))]; lab = "LCB"
        end
        yn = ev!(xn); X = hcat(X, xn); push!(y, yn)
        if yn < by
            by = yn; bx = copy(xn); stag = 0
            println("    eval=$(n_init+iter) [$lab] ✓ ar=$(round(-by/n_edges,digits=4))")
            flush(stdout)
        else; t >= phase1_end && (stag += 1); end
        curve[eval_idx] = by
        if stag >= stag_limit
            for _ in 1:2
                xr = lo .+ rand(dim) .* (hi .- lo); yr = ev!(xr)
                X = hcat(X, xr); push!(y, yr); yr < by && (by = yr; bx = copy(xr))
            end
            stag = 0; curve[eval_idx] = by
        end
        if iter % refine_every == 0 && t >= phase1_end
            for sc in [0.05, 0.20, 0.50, 0.80, 1.0]
                xs0 = clamp.(bx .+ randn(dim) .* sc .* (hi .- lo), lo .+ 1e-6, hi .- 1e-6)
                r   = optimize(x -> ev!(clamp.(x,lo,hi)), xs0, NelderMead(),
                               Optim.Options(iterations=50, f_reltol=1e-6, show_trace=false))
                yr  = Optim.minimum(r); xr = clamp.(Optim.minimizer(r), lo, hi)
                if yr < by - 1e-6
                    println("    [MRLR sc=$sc] ✓ $(round(-by/n_edges,digits=4)) → $(round(-yr/n_edges,digits=4))")
                    flush(stdout)
                    by = yr; bx = xr; stag = 0; X = hcat(X, xr); push!(y, yr)
                end
            end
            curve[eval_idx] = by
        end
    end
    for i in 2:total_budget; curve[i] = min(curve[i], curve[i-1]); end

    ar = -by / n_edges
    println("    done  ar=$(round(ar,digits=4))  evals=$(eval_count[])  time=$(round(time()-t_start,digits=1))s")
    flush(stdout)
    return (d, seed, by, bx, curve, eval_count[], n_edges, ar)
end



#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── JOINT 7D BO TASK FUNCTION ────────────────────────────────────────────────
# Optimises d AND (γ,β) jointly in a 7-dimensional space.
# x[1] = d (μm), x[2:7] = (γ1,β1,γ2,β2,γ3,β3)
# This eliminates the separate Phase 1/Phase 2 architecture and the
# boundary-clamping problem in the 1D spacing BO.

@everywhere function compute_joint_task(params)
    task_id = params.task_id
    seed    = SEEDS_W[((task_id - 1) % length(SEEDS_W)) + 1]
    budget  = params.budget
    n_init  = params.n_init

    lo = copy(LOWER_JOINT); hi = copy(UPPER_JOINT); dim = DIM_JOINT
    Random.seed!(seed)
    println("Starting joint task $(lpad(task_id,2))  seed=$(seed)  d∈[$(lo[1]),$(hi[1])] — $(Dates.format(now(),"HH:MM:SS"))")
    flush(stdout)

    cache = Dict{Vector{Float64},Float64}(); eval_count = Ref(0)
    function ev!(x)
        xs  = clamp.(x, lo, hi); key = round.(xs, digits=6)
        haskey(cache, key) && return cache[key]
        eval_count[] += 1
        d_val = xs[1]; angles = xs[2:end]
        try
            atoms = make_cube_atoms(d_val)
            edges = build_edges_by_radius(atoms, d_val * 1.2)
            lv    = loss_fn(angles, atoms, edges, length(atoms))
            if isfinite(lv); cache[key] = lv; return lv; end
        catch e; println("  eval error: $e"); end
        return 1000.0
    end

    # Halton warm-start in 7D
    halton_pts = halton_samples(n_init, lo, hi; skip=seed % 50)
    init_xs    = [clamp.(pt .+ randn(dim) .* 0.05 .* (hi .- lo), lo, hi)
                  for pt in halton_pts]
    init_ys    = Float64[ev!(x) for x in init_xs]
    X = hcat(init_xs...); y = copy(init_ys)
    by = minimum(init_ys); bx = init_xs[argmin(init_ys)]

    n_bo_iter = budget - n_init; total_budget = budget
    curve = fill(by, total_budget)
    for i in 1:n_init; curve[i] = minimum(init_ys[1:i]); end
    println("  warm-start done  d=$(round(bx[1],digits=2))μm  ar=$(round(-by/N_EDGES,digits=4))")

    Q_g=2; Q_c=2; map_every=20; n_cand=500
    phase1_end=0.25; phase2_end=0.65; stag_limit=30; refine_every=20
    θ = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=2); stag = 0

    for iter in 1:n_bo_iter
        t = iter/n_bo_iter; eval_idx = min(n_init+iter, total_budget)
        iter>1 && iter%map_every==0 && (θ = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=2))
        gp = build_spectral_gp(X, y, θ, Q_g, Q_c, dim)
        C  = lo .+ rand(dim, n_cand) .* (hi .- lo)
        if t < phase1_end
            bi = ts_acq(gp, C; ns=10); xn = C[:,bi]; lab = "TS"
        elseif t < phase2_end
            ξ   = 0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ = gp_pred(gp, C); xn = C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]; lab = "EI"
        else
            β   = max(exp(-2.0*(t-phase2_end)/(1-phase2_end)), 0.1)
            μ,σ = gp_pred(gp, C); xn = C[:,argmin(lcb_acq(μ,σ;β=β))]; lab = "LCB"
        end
        yn = ev!(xn); X = hcat(X, xn); push!(y, yn)
        if yn < by
            by = yn; bx = copy(xn); stag = 0
            println("    eval=$(n_init+iter) [$lab] ✓ d=$(round(bx[1],digits=2))μm ar=$(round(-by/N_EDGES,digits=4))")
            flush(stdout)
        else; t >= phase1_end && (stag += 1); end
        curve[eval_idx] = by
        if stag >= stag_limit
            for _ in 1:2
                xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                X=hcat(X,xr); push!(y,yr); yr<by && (by=yr; bx=copy(xr))
            end
            stag=0; curve[eval_idx]=by
        end
        if iter%refine_every==0 && t>=phase1_end
            for sc in [0.05, 0.20, 0.50]
                xs0 = clamp.(bx .+ randn(dim).*sc.*(hi.-lo), lo.+1e-6, hi.-1e-6)
                r   = optimize(x->ev!(clamp.(x,lo,hi)), xs0, NelderMead(),
                               Optim.Options(iterations=50, f_reltol=1e-6, show_trace=false))
                yr  = Optim.minimum(r); xr = clamp.(Optim.minimizer(r), lo, hi)
                if yr < by - 1e-6
                    println("    [MRLR sc=$sc] ✓ d=$(round(xr[1],digits=2))μm $(round(-by/N_EDGES,digits=4)) → $(round(-yr/N_EDGES,digits=4))")
                    flush(stdout)
                    by=yr; bx=xr; stag=0; X=hcat(X,xr); push!(y,yr)
                end
            end
            curve[eval_idx] = by
        end
    end
    for i in 2:total_budget; curve[i]=min(curve[i],curve[i-1]); end

    d_best = bx[1]; angles_best = bx[2:end]
    ar = -by/N_EDGES; elapsed = round(time()-params.t_start, digits=1)
    println("Finished joint task $(lpad(task_id,2))  d=$(round(d_best,digits=3))μm  ar=$(round(ar,digits=4))  evals=$(eval_count[])  time=$(elapsed)s")
    flush(stdout)
    return (task_id, d_best, seed, by, angles_best, curve, eval_count[], N_EDGES, ar, elapsed)
end


#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── P-SCALING WORKER FUNCTION ───────────────────────────────────────────────
# Runs the full GSM+CSM BO at a specified circuit depth p (not hardcoded P_VAL).
# params fields: task_id, p_val, d_fixed, budget, n_init, t_start
# d_fixed: atomic spacing in μm (use best known d* from Phase 1/2).
# Returns the same tuple structure as compute_task for easy aggregation.

@everywhere function compute_task_p(params)
    p_val    = params.p_val
    seed     = SEEDS_W[((params.task_id - 1) % length(SEEDS_W)) + 1]
    d        = params.d_fixed
    budget   = params.budget
    n_init   = params.n_init
    x0_hint  = get(params, :x0_hint, nothing)   # optional warm-start hint from p-1

    dim = 2 * p_val
    lo  = zeros(dim)
    hi  = vcat([[3π, π] for _ in 1:p_val]...)     # γ∈[0,3π], β∈[0,π] per layer

    Random.seed!(seed)
    println("  [p=$(p_val)] task $(lpad(params.task_id,2))  seed=$(seed)  d=$(d)μm  dim=$(dim) — " * Dates.format(now(),"HH:MM:SS"))
    flush(stdout)

    atoms   = make_cube_atoms(d)
    edges   = build_edges_by_radius(atoms, d * 1.2)
    n_atoms = length(atoms); n_edges = length(edges)

    cache = Dict{Vector{Float64},Float64}(); eval_count = Ref(0)
    function ev!(x)
        xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
        haskey(cache, key) && return cache[key]
        eval_count[] += 1
        try
            lv = loss_fn(xs, atoms, edges, n_atoms)
            if isfinite(lv); cache[key] = lv; return lv; end
        catch e; println("  eval error: $e"); end
        return 1000.0
    end

    n_h = min(n_init, length(HALTON_BASES) > dim ? n_init : n_init)
    halton_pts = halton_samples(n_h, lo, hi; skip=seed % 50)
    init_xs    = Vector{Vector{Float64}}(undef, n_h)
    # If x0_hint provided, inject jittered hint into first 30% of warm-start
    if !isnothing(x0_hint) && length(x0_hint) == dim
        n_hint = max(3, n_h ÷ 3)
        for i in 1:n_hint
            sc = 0.05 + 0.10*(i-1)/n_hint
            init_xs[i] = clamp.(x0_hint .+ randn(dim) .* sc .* (hi.-lo), lo, hi)
        end
        for i in (n_hint+1):n_h
            init_xs[i] = clamp.(halton_pts[i] .+ randn(dim) .* 0.05 .* (hi.-lo), lo, hi)
        end
    else
        for i in 1:n_h
            init_xs[i] = clamp.(halton_pts[i] .+ randn(dim) .* 0.05 .* (hi.-lo), lo, hi)
        end
    end
    init_ys    = Float64[ev!(x) for x in init_xs]
    X = hcat(init_xs...); y = copy(init_ys)
    by = minimum(init_ys); bx = init_xs[argmin(init_ys)]

    n_bo_iter = budget - n_init; total_budget = budget
    curve = fill(by, total_budget)
    for i in 1:n_init; curve[i] = minimum(init_ys[1:i]); end
    println("    warm-start  ar=$(round(-by/n_edges,digits=4))")

    # BO strategy (same as compute_task_p — proven by Q3_local_d14.jl)
    Q_g=2; Q_c=2
    n_cand       = dim >= 8 ? 800  : 400
    map_every    = dim >= 8 ? 12   : 18
    phase1_end   = dim >= 8 ? 0.45 : 0.30
    phase2_end   = dim >= 8 ? 0.78 : 0.68
    nr_hp        = dim >= 8 ? 4    : 2
    stag_limit   = dim >= 8 ? 25   : 20
    refine_every = 15
    nm_iters     = dim >= 8 ? 200  : 100
    refine_scales = dim >= 8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
    θ = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=nr_hp); stag = 0

    for iter in 1:n_bo_iter
        t = iter/n_bo_iter; eval_idx = min(n_init+iter, total_budget)
        iter>1 && iter%map_every==0 && (θ = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=nr_hp))
        gp = build_spectral_gp(X, y, θ, Q_g, Q_c, dim)
        C  = lo .+ rand(dim, n_cand) .* (hi .- lo)
        if t < phase1_end
            bi=ts_acq(gp,C;ns=10); xn=C[:,bi]; lab="TS"
        elseif t < phase2_end
            ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]; lab="EI"
        else
            β=max(exp(-2.0*(t-phase2_end)/(1-phase2_end)),0.1)
            μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]; lab="LCB"
        end
        yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
        if yn < by
            by=yn; bx=copy(xn); stag=0
            println("    eval=$(n_init+iter) [$lab] ✓ ar=$(round(-by/n_edges,digits=4))")
            flush(stdout)
        else; t>=phase1_end && (stag+=1); end
        curve[eval_idx]=by
        if stag>=stag_limit
            for _ in 1:2
                xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                X=hcat(X,xr); push!(y,yr); yr<by && (by=yr; bx=copy(xr))
            end
            stag=0; curve[eval_idx]=by
        end
        if iter%refine_every==0 && t>=phase1_end
            for sc in refine_scales
                xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                           Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                if yr<by-1e-6
                    by=yr; bx=xr; stag=0; X=hcat(X,xr); push!(y,yr)
                    println("    [MRLR sc=$(sc)] ✓ $(round(-by/n_edges,digits=4))")
                end
            end
            curve[eval_idx]=by
        end
    end
    for i in 2:total_budget; curve[i]=min(curve[i],curve[i-1]); end

    ar=-by/n_edges; elapsed=round(time()-params.t_start,digits=1)
    println("    done  ar=$(round(ar,digits=4))  evals=$(eval_count[])  $(elapsed)s")
    flush(stdout)
    return (params.task_id, p_val, d, seed, by, bx, curve, eval_count[], n_edges, ar, elapsed)
end

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── BASELINE COMPARISON WORKER FUNCTION ─────────────────────────────────────
# Runs a classical optimiser (COBYLA, L-BFGS-B, NelderMead, random) on the
# identical problem as compute_task, at the identical budget.
# params fields: task_id, method (Symbol), p_val, d_fixed, budget, n_init, t_start
# All methods use the same ev!() function, d*, and budget for a fair comparison.

@everywhere function compute_baseline_task(params)
    method     = params.method          # :cobyla | :spsa | :neldermead | :random
    p_val      = params.p_val
    seed       = SEEDS_W[((params.task_id - 1) % length(SEEDS_W)) + 1]
    d          = params.d_fixed
    budget     = params.budget
    graph_type = get(params, :graph_type, "q3")   # [FIX] 默认 q3, 向后兼容旧调用方式
    n_at       = get(params, :n_at, 8)

    dim = 2 * p_val
    lo  = zeros(dim)
    hi  = vcat([[3π, π] for _ in 1:p_val]...)

    Random.seed!(seed)
    # [FIX] baseline 模式在 make_graph 定义之前就 pmap 了 → worker 上 UndefVarError。
    # q3 直接用 BO 同一套函数建图, 保证对照的是同一张图。
    atoms, edges, n_atoms, n_edges = if graph_type == "q3"
        _at = make_cube_atoms(d); _ed = build_edges_by_radius(_at, d * 1.2)
        (_at, _ed, length(_at), length(_ed))
    else
        _r = make_graph(graph_type, n_at, d; seed=42); (_r[1], _r[2], _r[3], _r[4])
    end

    eval_count = Ref(0)
    curve = fill(1.0, budget)    # track best-so-far loss (will negate for ar)

    function ev_tracked!(x)
        eval_count[] >= budget && return 1000.0   # hard cap at budget
        xs = clamp.(x, lo, hi)
        eval_count[] += 1
        try
            lv = loss_fn(xs, atoms, edges, n_atoms)
            # Clamp to [-n_edges-1, 1000]: valid loss ∈ [-n_edges, 0].
            # Without this, L-BFGS-B divergence produces huge negative values
            # that pollute `by` and give ar = -83 (bug observed in practice).
            v  = isfinite(lv) ? clamp(lv, -(n_edges+1.0), 1000.0) : 1000.0
            idx = min(eval_count[], budget)
            curve[idx] = v
            return v
        catch
            return 1000.0
        end
    end

    by = 1000.0; bx = lo .+ 0.5 .* (hi .- lo)

    if method == :random
        # ── Pure random search ─────────────────────────────────────────────
        println("  [baseline:random  p=$(p_val)] seed=$(seed) — " * Dates.format(now(),"HH:MM:SS"))
        for _ in 1:budget
            eval_count[] >= budget && break
            x = lo .+ rand(dim) .* (hi .- lo)
            y = ev_tracked!(x)
            y < by && (by=y; bx=copy(x))
        end

    elseif method == :neldermead
        # ── Nelder-Mead without BO: pure direct-search ────────────────────
        println("  [baseline:neldermead p=$(p_val)] seed=$(seed) — " * Dates.format(now(),"HH:MM:SS"))
        # Multi-start: n_start restarts of NelderMead, each gets budget/n_start evals
        n_start = 5; evals_per_start = budget ÷ n_start
        for _ in 1:(get(ENV, "QAOA_BASELINE_FILL", "0") == "1" ? typemax(Int) : n_start)
            eval_count[] >= budget && break
            x0 = lo .+ rand(dim) .* (hi .- lo)
            r  = optimize(ev_tracked!, lo, hi, x0, Fminbox(NelderMead()),
                          Optim.Options(iterations=evals_per_start÷(dim+2),
                                        show_trace=false))
            yr = Optim.minimum(r); xr = clamp.(Optim.minimizer(r), lo, hi)
            yr < by && (by=yr; bx=xr)
        end

    elseif method == :spsa
        # ── SPSA (Simultaneous Perturbation Stochastic Approximation) ──────
        # Replaces L-BFGS-B which fails on this non-smooth bounded objective:
        # Fminbox+LBFGS uses barrier functions + finite-diff gradients, but
        # clamping in ev_tracked! makes f(x)=f(x+h) at boundaries → gradient≈0
        # → converges at starting point → yr=1000 → by never updates → ar=-83.
        #
        # SPSA uses 2 random ±ck perturbations to estimate the gradient,
        # making it robust to non-smoothness and boundary effects.
        # It is the standard gradient-free optimiser for quantum circuits
        # (used in Qiskit, PennyLane, and published QAOA benchmarks).
        println("  [baseline:spsa  p=$(p_val)] seed=$(seed) — " * Dates.format(now(),"HH:MM:SS"))
        # SPSA hyperparameters (Spall 1998 recommendations)
        a0 = 0.2 * (hi[1] - lo[1]); c0 = 0.1 * (hi[1] - lo[1])
        α  = 0.602; γ_sp = 0.101; A = 0.1 * budget
        x_sp = lo .+ rand(dim) .* (hi .- lo)   # random start
        for k in 1:budget÷2   # each SPSA iteration uses 2 evals
            eval_count[] >= budget && break
            ak = a0 / (k + A)^α
            ck = c0 / k^γ_sp
            Δ  = 2.0 .* (rand(dim) .> 0.5) .- 1.0   # Rademacher ±1
            xp = clamp.(x_sp .+ ck .* Δ, lo, hi)
            xm = clamp.(x_sp .- ck .* Δ, lo, hi)
            fp = ev_tracked!(xp); fm = ev_tracked!(xm)
            ghat = (fp - fm) ./ (2 .* ck .* Δ)
            x_sp = clamp.(x_sp .- ak .* ghat, lo, hi)
            v_sp = ev_tracked!(x_sp)
            eval_count[] > 1 && v_sp < by && (by = v_sp; bx = copy(x_sp))
        end

    elseif method == :cobyla
        # ── COBYLA (Constrained Optimisation BY Linear Approximations) ────
        # Optim.jl doesn't have COBYLA; use NelderMead with tight simplex
        # as the closest available derivative-free method in pure Julia.
        # For true COBYLA, Python's scipy.optimize.fmin_cobyla would be needed.
        # We implement a gradient-free trust-region style via many short NM runs.
        println("  [baseline:cobyla   p=$(p_val)] seed=$(seed) — " * Dates.format(now(),"HH:MM:SS"))
        n_start = 10; evals_per_start = budget ÷ n_start
        for _ in 1:(get(ENV, "QAOA_BASELINE_FILL", "0") == "1" ? typemax(Int) : n_start)
            eval_count[] >= budget && break
            x0 = lo .+ rand(dim) .* (hi .- lo)
            r  = optimize(ev_tracked!, lo, hi, x0, Fminbox(NelderMead()),
                          Optim.Options(iterations=evals_per_start÷(dim+2),
                                        show_trace=false))
            yr = Optim.minimum(r); xr = clamp.(Optim.minimizer(r), lo, hi)
            yr < by && (by=yr; bx=xr)
        end
    end

    # Forward-fill curve (best-so-far)
    best_seen = 1000.0
    for i in 1:budget
        curve[i] < best_seen && (best_seen = curve[i])
        curve[i] = best_seen
    end

    ar = -by/n_edges; elapsed = round(time()-params.t_start, digits=1)
    println("  done  method=$(method)  p=$(p_val)  ar=$(round(ar,digits=4))  evals=$(eval_count[])  $(elapsed)s")
    flush(stdout)
    return (params.task_id, method, p_val, d, seed, by, bx, curve, eval_count[], n_edges, ar, elapsed, graph_type)
end

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── OUTPUT DIRECTORY ────────────────────────────────────────────────────────
const JOB_LABEL = "Q3_QAOA_GSM_CSM"

# [FIX] 原来 SLURM 分支写死 /scratch/$USER —— 很多集群根本没有 /scratch
# (可能是 /lustre、/work、/gpfs、/home 等), 目录建不出来就意味着**所有结果
# 文件写入失败**, 几小时的作业白跑而且直到最后才发现。原来 SLURM 的判据用的
# 是 SLURM_JOBID, 而核数检测用的是 SLURM_JOB_ID —— 两个变量名不同, 有的集群
# 只设其中一个, 会出现"核数认出是SLURM但输出目录走了本地分支"的错配。
#
# 现在改成: 优先用 QAOA_OUTDIR 环境变量(集群上直接指到你有写权限的目录),
# 没设的话一律用**当前工作目录**下的子目录 —— 作业脚本里 cd 到提交目录,
# 那里必然可写, 比猜一个可能不存在的全局路径可靠得多。
if haskey(ENV, "QAOA_OUTDIR")
    directoryBasis = ENV["QAOA_OUTDIR"]
elseif haskey(ENV, "SGE_CELL") || haskey(ENV, "JOB_ID")
    directoryBasis = joinpath(pwd(), "SGE-$(JOB_LABEL)")
elseif haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
    directoryBasis = joinpath(pwd(), "SLURM-$(JOB_LABEL)")
elseif haskey(ENV, "PBS_JOBID")
    directoryBasis = joinpath(pwd(), "PBS-$(JOB_LABEL)")
else
    directoryBasis = joinpath(pwd(), "Local-$(JOB_LABEL)")
end

# 启动时就验证目录可建、可写 —— 宁可现在立刻失败, 也不要跑几小时后才发现
# 结果写不进去。集群上路径/配额/权限问题很常见, 这个自检成本几乎为零。
try
    mkpath(directoryBasis)
    let probe = joinpath(directoryBasis, ".write_test_$(getpid())")
        write(probe, "ok"); rm(probe, force=true)
    end
    println("输出目录: $(directoryBasis)  (可写 ✓)")
    global directorySteps = joinpath(directoryBasis, "ProgressSteps")
    mkpath(directorySteps)
    println("逐步进度: $(directorySteps)  (每完成一步落一个文件, ls | wc -l 即可看进度)")
catch e
    println("=" ^ 70)
    println("致命: 输出目录不可写 → $(directoryBasis)")
    println("所有结果(CSV/追踪/checkpoint/先验)都会写不进去, 现在中止而不是跑完才发现。")
    println("解决: 设 QAOA_OUTDIR 指到一个你有写权限的目录, 例如")
    println("  export QAOA_OUTDIR=\$SLURM_SUBMIT_DIR/qaoa_out")
    println("=" ^ 70)
    rethrow(e)
end

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── PHASE 2 + SUMMARY (runs on master after pmap) ───────────────────────────

function run_phase2_and_summary(all_d, all_ar, all_bx, all_curves, all_seeds_run,
                                 init_layer::InitLayer, n_bo_iter_d::Int)
    U_vals(d) = C6_VAL / (4 * d^6)

    if n_bo_iter_d > 0 && length(all_d) >= 2
        println("\n[Phase 2] Spacing BO ($n_bo_iter_d iterations)...")
        println("="^70)
        # Search range extends BEYOND D_CANDIDATES so Phase 2 never
        # starts at an existing observation boundary.
        # ±2.0 μm buffer beyond D_CANDIDATES prevents boundary-clamping loops.
        d_lo = minimum(D_CANDIDATES) - 2.0   # = 9.5 μm with new grid
        d_hi = maximum(D_CANDIDATES) + 2.0   # = 16.0 μm with new grid
        best_so_far = maximum(all_ar)

        for iter in 1:n_bo_iter_d
            d_grid = range(d_lo, d_hi, length=300) |> collect
            μ, σ   = spacing_gp_predict(all_d, -all_ar, d_grid)
            acq    = ei_spacing(μ, σ, minimum(-all_ar))
            d_next = d_grid[argmax(acq)]
            # Enforce minimum gap of 0.4 μm from any existing d.
            # Loop until we find a genuinely novel spacing.
            _attempts = 0
            while minimum(abs.(all_d .- d_next)) < 0.4 && _attempts < 20
                _attempts += 1
                d_next = clamp(d_next + randn()*2.0, d_lo, d_hi)
            end
            # Fallback: find the largest gap between sorted existing d-values
            # and sample from its midpoint — guarantees a novel d each time.
            if minimum(abs.(all_d .- d_next)) < 0.4
                d_sorted  = sort(unique(round.(all_d, digits=2)))
                bounds    = vcat([d_lo], d_sorted, [d_hi])
                gaps      = diff(bounds)
                best_i    = argmax(gaps)
                d_next    = bounds[best_i] + 0.5 * gaps[best_i]  # midpoint of widest gap
                println("  [Phase 2 fallback] largest-gap → d=$(round(d_next,digits=2)) μm")
            end

            @printf("[Phase 2 iter=%d] d=%.2f μm...\n", iter, d_next)
            flush(stdout)

            b2 = -Inf; bx2 = zeros(DIM); crv2 = zeros(QAOA_BUDGET)
            x0h = predict_x0(init_layer, d_next)   # Layer 2 hint for d_next

            # Call compute_phase2_task with the actual d_next — fixes the critical bug
            # where the old code ignored d_next and used fixed D_CANDIDATES values.
            seed_results = []
            for seed in SEEDS
                res = compute_phase2_task(d_next, seed,
                                          QAOA_BUDGET, QAOA_N_INIT,
                                          x0h, time())
                (_, _, by2, bx2t, crv2t, _, n_ed, ar2) = res
                @printf("  seed=%d  ar=%.4f\n", seed, ar2)
                flush(stdout)
                push!(seed_results, res)
                ar2 > b2 && (b2 = ar2; bx2 = bx2t; crv2 = crv2t)
            end

            # Push best result for summary/Layer2 bookkeeping
            push!(all_d, d_next); push!(all_ar, b2)
            push!(all_bx, bx2);   push!(all_curves, crv2)
            push!(all_seeds_run, length(SEEDS))
            update_init_layer!(init_layer, d_next, bx2)
            # Also add the other seed results as noisy GP observations
            # so the spacing GP sees the full variance, not just the best.
            for (_, _s, _by2, _bx2t, _crv2t, _, _ned, _ar2) in seed_results
                if _ar2 != b2   # don't double-add the best
                    push!(all_d, d_next); push!(all_ar, _ar2)
                    push!(all_bx, _bx2t); push!(all_curves, _crv2t)
                    push!(all_seeds_run, 1)
                end
            end
            b2 > best_so_far &&
                (best_so_far = b2;
                 @printf("  ★ New best! ar=%.4f at d=%.2f μm\n", b2, d_next))
        end
    end

    # ── Final summary table ──────────────────────────────────────────────────
    idx_sorted = sortperm(all_ar, rev=true)
    best_i     = idx_sorted[1]

    println("\n\n╔" * "═"^76 * "╗")
    println("║   Final Results                                                            ║")
    println("╠" * "═"^76 * "╣")
    println("║  Rank  │  d (μm)  │  approx ratio  │  cut value  │  U_ij          ║")
    println("╠" * "═"^76 * "╣")
    for (rank, i) in enumerate(idx_sorted[1:min(10,end)])
        @printf("║  %4d  │  %6.2f  │    %.4f      │   %7.4f   │  %12.4f  ║\n",
            rank, all_d[i], all_ar[i], (all_ar[i]*N_EDGES), U_vals(all_d[i]))
    end
    println("╠" * "═"^76 * "╣")
    @printf("║  Global best:  d=%.2f μm,  ar=%.4f,  cut=%.4f\n",
        all_d[best_i], all_ar[best_i], (all_ar[best_i]*N_EDGES))
    println("║  Best params γ,β = $(round.(all_bx[best_i], digits=4))")
    println("║  Reference: GW approx ratio ≈ 0.878,  max cut = $N_EDGES (Q3 hypercube)")
    println("╚" * "═"^76 * "╝")

    # ── Convergence table ────────────────────────────────────────────────────
    println("\n=== Convergence (Phase 1 candidates, every 25 steps) ===")
    hdr = "  eval │"
    for d in D_CANDIDATES; hdr *= @sprintf(" d=%-4.0f │", d); end
    println(hdr)
    for ci in vcat(collect(1:25:QAOA_BUDGET), [QAOA_BUDGET])
        row = @sprintf("  %4d │", ci)
        for dc in D_CANDIDATES
            idx = findfirst(==(dc), all_d)
            val = idx !== nothing ? -all_curves[idx][min(ci,end)] / N_EDGES : NaN
            row *= @sprintf(" %6.4f │", val)
        end
        println(row)
    end

    # ── U_ij vs ar table ─────────────────────────────────────────────────────
    println("\n=== U_ij vs approximation ratio ===")
    println("  U_ij         │  approx ratio  │  seeds  │  trend")
    ar_p1 = [all_ar[findfirst(==(d), all_d)] for d in D_CANDIDATES]
    ns_p1 = [all_seeds_run[findfirst(==(d), all_d)] for d in D_CANDIDATES]
    for (i, d) in enumerate(D_CANDIDATES)
        trend = i == 1 ? "" : ar_p1[i] > ar_p1[i-1] ? "↑" : "↓"
        @printf("  %12.2f │    %.4f     │    %d    │  %s\n",
            U_vals(d), ar_p1[i], ns_p1[i], trend)
    end
end


#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── ASSEMBLE PARAMETER LIST ─────────────────────────────────────────────────
# One NamedTuple per task — mirrors the template's all_params pattern.

t_global_start = time()

all_params = [
    (task_id = tid,
     budget  = QAOA_BUDGET,
     n_init  = QAOA_N_INIT,
     t_start = t_global_start)
    for tid in 1:N_TASKS
]

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── JOINT mode ──────────────────────────────────────────────────────────────
# ── SETUP mode — print SGE submit scripts ────────────────────────────────────
# julia Q3_sge.jl setup
if length(ARGS) >= 1 && ARGS[1] == "setup"
    project_path = Base.active_project()
    script_path  = abspath(@__FILE__)
    home         = homedir()
    println("""
#=== SGE SUBMIT SCRIPTS FOR Q3_QAOA_GSM_CSM ===
# Copy each block to a .sh file, then submit with: qsub <file>.sh

#--- 1. ADAPTIVE P-SCALING CHAIN (recommended: p=1→5, ~6h on 20 slots) ---
cat > submit_chain.sh << 'EOF'
#!/bin/bash
#\$ -N Q3_chain
#\$ -pe smp 100
#\$ -l h_rt=12:00:00
#\$ -l mem=4G
#\$ -cwd
#\$ -j y
#\$ -o chain_\$JOB_ID.log
export JULIA_CONDAPKG_BACKEND=Null
julia --project=$(project_path) $(script_path) pscale_chain 14.0 0.5 5
EOF
qsub submit_chain.sh

#--- 2. DEFAULT SWEEP (Phase 1 + Phase 2, d=[12.5..15.0]) ---
cat > submit_sweep.sh << 'EOF'
#!/bin/bash
#\$ -N Q3_sweep
#\$ -pe smp 20
#\$ -l h_rt=12:00:00
#\$ -l mem=4G
#\$ -cwd
#\$ -j y
#\$ -o sweep_\$JOB_ID.log
export JULIA_CONDAPKG_BACKEND=Null
julia --project=$(project_path) $(script_path)
EOF
qsub submit_sweep.sh

#--- 3. BASELINES (p=3 at d=14.0, all 4 methods) ---
cat > submit_baselines.sh << 'EOF'
#!/bin/bash
#\$ -N Q3_baselines
#\$ -pe smp 20
#\$ -l h_rt=6:00:00
#\$ -l mem=4G
#\$ -cwd
#\$ -j y
#\$ -o baselines_\$JOB_ID.log
export JULIA_CONDAPKG_BACKEND=Null
julia --project=$(project_path) $(script_path) baseline random 3 14.0
julia --project=$(project_path) $(script_path) baseline neldermead 3 14.0
julia --project=$(project_path) $(script_path) baseline spsa 3 14.0
julia --project=$(project_path) $(script_path) baseline cobyla 3 14.0
EOF
qsub submit_baselines.sh
""")
    exit(0)
end

# ── PSCALE_CHAIN mode ────────────────────────────────────────────────────────
# julia Q3_sge.jl pscale_chain [d_init] [d_step] [n_grid]
# Runs p=1→5 sequentially. After each p, the d-value that achieved the best ar
# becomes the grid centre for the next p. This ensures each p is optimised at
# the spacing that is optimal for that circuit depth.
#
# d_init : starting grid centre (default 14.0 μm — current best known d*)
# d_step : spacing between grid points (default 0.5 μm)
# n_grid : number of d-values in grid, should be odd for symmetric grid (default 5)
if length(ARGS) >= 1 && ARGS[1] == "pscale_chain"
    d_centre = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 14.0
    d_step   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5
    n_grid   = length(ARGS) >= 4 ? parse(Int,     ARGS[4]) : 5
    p_max    = length(ARGS) >= 5 ? parse(Int,     ARGS[5]) : 5
    budget_cap = parse(Int, get(ENV, "QAOA_CHAIN_BUDGET_CAP", "1000"))
    chain_resume = get(ENV, "QAOA_CHAIN_RESUME", "1") == "1"
    using Statistics

    println("\n" * "="^70)
    println("ADAPTIVE P-SCALING CHAIN: p=1→$(p_max)  (预算上限 $(budget_cap), 续跑=$(chain_resume))")
    println("Initial d centre: $(d_centre) μm  step: $(d_step) μm  grid: $(n_grid) points")
    println("="^70)

    chain_results = Dict{Int, Any}()   # p => (best_ar, best_d, best_params, all_results)
    mkpath(directoryBasis)

    # Use let-block so d_centre is a single mutable variable visible inside
    # the for-loop. Without this, Julia's soft-scope rules at top level
    # treat the assignment inside the loop as a new local → UndefVarError.
    let d_centre = d_centre, d_step = d_step, n_grid = n_grid
    for p_val in 1:p_max
        prev_files = filter(fn -> startswith(fn, "chain_p$(p_val)_d") && endswith(fn, ".jls"), readdir(directoryBasis))
        if chain_resume && !isempty(prev_files)
            p_results = deserialize(joinpath(directoryBasis, prev_files[end]))
            best_r = argmax(r -> r[10], p_results)
            chain_results[p_val] = (best_ar=best_r[10], best_d=best_r[3], best_params=best_r[6], all_results=p_results)
            d_centre = best_r[3]
            println("\n── p=$(p_val): 读入已有结果 $(prev_files[end]), best ar=$(round(best_r[10], digits=4)), 跳过")
            continue
        end
        # Build symmetric d-grid around current best centre
        half = (n_grid - 1) ÷ 2
        d_grid_chain = [d_centre + i*d_step for i in -half:half]
        # Clamp to physically reasonable range [9.0, 20.0]
        d_grid_chain = clamp.(d_grid_chain, 9.0, 20.0)
        d_grid_chain = sort(unique(round.(d_grid_chain, digits=3)))

        dim_p    = 2 * p_val
        # Budget: 100 evals/dim — proven by Q3_local_d14.jl results.
        # p=1:200, p=2:400, p=3:600, p=4:800, p=5:1000.
        ps_bud   = min(max(QAOA_BUDGET, 100 * dim_p), budget_cap)
        ps_init  = max(QAOA_N_INIT,   8 * dim_p)  # 8 warm-start per dim

        println("\n── p=$(p_val)  d-grid=$(d_grid_chain)  budget=$(ps_bud)  dim=$(dim_p)D ──")
        flush(stdout)

        # Dispatch: one task per (d, seed) combination
        # x0_hint: carry best params from p-1 into warm-start.
        # Extend γ/β array by appending the mean γ and mean β of the
        # previous best, so the new layer starts near the optimal values.
        x0_hint_chain = nothing
        if p_val > 1 && haskey(chain_results, p_val-1)
            prev_bx  = chain_results[p_val-1].best_params
            prev_dim = length(prev_bx)
            if prev_dim == 2*(p_val-1)  # sanity check
                γ_mean = mean(prev_bx[1:2:prev_dim])
                β_mean = mean(prev_bx[2:2:prev_dim])
                x0_hint_chain = vcat(prev_bx, [γ_mean, β_mean])
                println("  x0_hint: extended from p=$(p_val-1) dim=$(prev_dim)→$(dim_p)")
            end
        end
        all_p_params = [(task_id   = (di-1)*length(SEEDS) + si,
                         p_val     = p_val,
                         d_fixed   = d_grid_chain[di],
                         budget    = ps_bud,
                         n_init    = ps_init,
                         x0_hint   = x0_hint_chain,
                         t_start   = time())
                        for di in 1:length(d_grid_chain)
                        for si in 1:length(SEEDS)]

        @time p_results = pmap(compute_task_p, all_p_params)

        # Find best (d, seed) by ar
        best_r   = argmax(r -> r[10], p_results)
        best_ar  = best_r[10]
        best_d   = best_r[3]
        best_bx  = best_r[6]

        # Per-d statistics
        println("\n  Results p=$(p_val):")
        println("  " * "─"^60)
        for d_val in d_grid_chain
            d_ars = [r[10] for r in p_results if r[3] ≈ d_val]
            isempty(d_ars) && continue
            marker = d_val ≈ best_d ? " ← best d*" : ""
            @printf("  d=%5.2f μm: mean=%.4f  std=%.4f  best=%.4f  (n=%d)%s\n",
                d_val, mean(d_ars), std(d_ars), maximum(d_ars), length(d_ars), marker)
        end
        println("  " * "─"^60)
        @printf("  p=%d best: d*=%.3f μm  ar=%.4f  γ,β=%s\n",
            p_val, best_d, best_ar, string(round.(best_bx, digits=4)))
        flush(stdout)

        # Save results
        tag = "chain_p$(p_val)_d$(round(best_d,digits=2))"
        serialize(joinpath(directoryBasis,"$(tag).jls"), p_results)
        println("  ✓ Saved → " * joinpath(directoryBasis, tag * ".jls"))

        chain_results[p_val] = (best_ar=best_ar, best_d=best_d, best_params=best_bx,
                                 all_results=p_results)

        # Update d_centre for next p using best d* from this p
        d_centre = best_d
        println("  → next p=$(p_val+1) grid centred at d=$(round(d_centre,digits=3)) μm")
    end

    # ── Final summary table ──────────────────────────────────────────────
    end  # let d_centre

    println("\n\n" * "═"^70)
    println("ADAPTIVE P-SCALING CHAIN — FINAL SUMMARY")
    println("═"^70)
    println("║  p  │  d* (μm)  │  best ar  │  gap to GW  │  γ,β (first 2)")
    println("╠" * "─"^68)
    GW_bound = 0.878
    for p_val in 1:p_max
        haskey(chain_results, p_val) || continue
        r = chain_results[p_val]
        gap = GW_bound - r.best_ar
        params_short = string(round.(r.best_params[1:min(4,end)], digits=3))
        @printf("║  %d  │  %7.3f  │  %.4f   │  %.4f (%4.1f%%)  │  %s\n",
            p_val, r.best_d, r.best_ar, gap, gap/GW_bound*100, params_short)
    end
    println("╚" * "═"^68)
    exit(0)
end

# ── PSCALE mode ─────────────────────────────────────────────────────────────
# julia Q3_sge.jl pscale <p> [d_fixed]
# Runs GSM+CSM BO at circuit depth p with d fixed to d_fixed (default 13.0 μm)
# Uses all SEEDS (20) for publication-grade statistics.
if length(ARGS) >= 1 && ARGS[1] == "pscale"
    p_val   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    d_fixed = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 14.0  # best d* from Phase 1
    n_ps    = length(SEEDS)
    println("P-scaling mode: p=$(p_val)  d=$(d_fixed) μm  $(n_ps) seeds")
    println("Parameter space: $(2*p_val)D  γ∈[0,3π]×$(p_val), β∈[0,π]×$(p_val)")
    # Auto-scale budget with dimension: p=4 needs 8D space, p=5 needs 10D.
    # Rule: budget = max(QAOA_BUDGET, 50 * dim) where dim = 2*p_val.
    # p=1→200, p=2→200, p=3→200, p=4→400, p=5→500.
    # Budget formula: 100 evals/dim — proven by Q3_local_d14.jl
    # (p=4 mean=0.877 best=0.893, p=5 mean=0.929 best=0.950).
    # p=1:200, p=2:400, p=3:600, p=4:800, p=5:1000.
    ps_budget  = max(QAOA_BUDGET, 100 * 2 * p_val)
    ps_n_init  = max(QAOA_N_INIT,   8 * 2 * p_val)  # 8 warm-start per dim
    println("  budget=$(ps_budget)  n_init=$(ps_n_init)  (auto-scaled for $(2*p_val)D)")
    ps_params = [(task_id=tid, p_val=p_val, d_fixed=d_fixed,
                  budget=ps_budget, n_init=ps_n_init, t_start=time())
                 for tid in 1:n_ps]
    @time ps_results = pmap(compute_task_p, ps_params)
    ars = [r[10] for r in ps_results]
    println("\n=== p=$(p_val) results ($(n_ps) seeds) ===")
    println("╔"*"═"^60*"╗")
    println("║  seed │  ar      │  cut     │  evals  │  time   ║")
    println("╠"*"═"^60*"╣")
    for r in sort(ps_results, by=x->x[4])
        (tid,pv,d,seed,by,bx,curve,ne,nedge,ar,elapsed) = r
        @printf("║  %4d │  %.4f  │  %.4f  │  %5d  │ %6.0fs ║\n",
            seed, ar, -by, ne, elapsed)
    end
    println("╠"*"═"^60*"╣")
    using Statistics
    @printf("║  mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n",
        mean(ars), std(ars), minimum(ars), maximum(ars))
    println("║  P(ar>0.75)=$(round(mean(ars.>0.75),digits=3))"*
             "  P(ar>0.78)=$(round(mean(ars.>0.78),digits=3))")
    println("╚"*"═"^60*"╝")
    tag = "pscale_p$(p_val)_d$(d_fixed)"
    mkpath(directoryBasis)
    serialize(joinpath(directoryBasis,"$(tag).jls"), ps_results)
    println("✓ Saved → " * joinpath(directoryBasis, tag * ".jls"))
    exit(0)
end

# ── BASELINE mode ────────────────────────────────────────────────────────────
# julia Q3_sge.jl baseline <method> [p] [d_fixed]
# method: cobyla | spsa | neldermead | random   (spsa replaces lbfgsb — see compute_baseline_task)
# Runs the chosen classical optimiser at the same budget as the BO, same seeds.
if length(ARGS) >= 1 && ARGS[1] == "baseline"
    length(ARGS) < 2 && error("Usage: julia Q3_sge.jl baseline <method> [p] [d] [graph_type] [n_atoms]")
    method_str = ARGS[2]
    method_sym = Symbol(method_str)
    method_sym in (:cobyla, :spsa, :neldermead, :random) ||
        error("method must be: cobyla | spsa | neldermead | random")
    p_val      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    d_fixed    = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 13.0
    graph_type = length(ARGS) >= 5 ? ARGS[5] : "q3"            # [FIX] 新增: q3|q4|reg3|udisk
    n_at       = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 8   # [FIX] 新增: 任意原子数(reg3/udisk用)
    n_bl       = length(SEEDS)
    println("Baseline mode: method=$(method_str)  p=$(p_val)  d=$(d_fixed)μm  graph=$(graph_type)(n=$(n_at))  $(n_bl) seeds")
    bl_params = [(task_id=tid, method=method_sym, p_val=p_val, d_fixed=d_fixed,
                  budget=parse(Int, get(ENV, "QAOA_BASELINE_BUDGET", string(QAOA_BUDGET))), n_init=QAOA_N_INIT, t_start=time(),
                  graph_type=graph_type, n_at=n_at)
                 for tid in 1:n_bl]
    @time bl_results = pmap(compute_baseline_task, bl_params)
    ars = [r[11] for r in bl_results]
    println("\n=== Baseline $(method_str)  graph=$(graph_type)(n=$(n_at))  p=$(p_val) results ===")
    println("╔"*"═"^60*"╗")
    println("║  seed │  ar      │  cut     │  evals  │  time   ║")
    println("╠"*"═"^60*"╣")
    for r in sort(bl_results, by=x->x[5])
        (tid,meth,pv,d,seed,by,bx,curve,ne,nedge,ar,elapsed,gt) = r
        @printf("║  %4d │  %.4f  │  %.4f  │  %5d  │ %6.0fs ║\n",
            seed, ar, -by, ne, elapsed)
    end
    println("╠"*"═"^60*"╣")
    using Statistics
    @printf("║  mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n",
        mean(ars), std(ars), minimum(ars), maximum(ars))
    println("╚"*"═"^60*"╝")
    tag = "baseline_$(method_str)_$(graph_type)_n$(n_at)_p$(p_val)_d$(d_fixed)"
    mkpath(directoryBasis)
    serialize(joinpath(directoryBasis,"$(tag).jls"), bl_results)
    println("✓ Saved → " * joinpath(directoryBasis, tag * ".jls"))
    exit(0)
end

if length(ARGS) >= 1 && ARGS[1] == "joint"
    println("Joint 7D BO mode: $(N_JOINT_TASKS) tasks — d∈[$(LOWER_JOINT[1]),$(UPPER_JOINT[1])] μm + 6 angles")
    joint_params = [(task_id=tid, budget=QAOA_BUDGET,
                     n_init=QAOA_N_INIT, t_start=time())
                    for tid in 1:N_JOINT_TASKS]
    @time joint_results = pmap(compute_joint_task, joint_params)
    println("\n=== Joint 7D BO results ===")
    for r in sort(joint_results, by=x->x[1])
        (tid,d,seed,by,bx,curve,nevals,nedge,ar,elapsed) = r
        @printf("task %d: d=%.3f μm  ar=%.4f  evals=%d  time=%.0fs\n",
            tid, d, ar, nevals, elapsed)
    end
    best_j = argmax(r->r[9], joint_results)
    @printf("Global best: d=%.3f μm  ar=%.4f  γ,β=%s\n",
        best_j[2], best_j[9], string(round.(best_j[5], digits=4)))
    exit(0)
end

# ── VERIFY mode ──────────────────────────────────────────────────────────────
if length(ARGS) >= 1 && ARGS[1] == "verify"
    println("=== Edge verification (Q3 hypercube) ===")
    for d in D_CANDIDATES
        at=make_cube_atoms(d); ed=build_edges_by_radius(at,d*1.2)
        println("  d=$d μm: $(length(ed)) edges  $ed")
    end
    exit(0)
end

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── AGGREGATE mode ───────────────────────────────────────────────────────────
if length(ARGS) >= 1 && ARGS[1] == "aggregate"
    results_dir = length(ARGS)>=2 ? ARGS[2] : directoryBasis
    n_bo = length(ARGS)>=3 ? parse(Int,ARGS[3]) : N_BO_ITER_D
    println("Aggregate mode: loading from $results_dir")
    init_layer=InitLayer(); all_d=Float64[]; all_ar=Float64[]
    all_bx=Vector{Vector{Float64}}(); all_curves=Vector{Vector{Float64}}()
    all_seeds_run=Int[]
    for (di,d) in enumerate(D_CANDIDATES)
        d_res=[]
        for (si,seed) in enumerate(SEEDS)
            tid=(di-1)*length(SEEDS)+si
            f=joinpath(results_dir,"task_$(lpad(tid,3,'0')).jls")
            isfile(f)&&push!(d_res,deserialize(f))
        end
        isempty(d_res)&&continue
        best=argmax(r->r[9],d_res)   # argmax(f,col) returns element, not index
        push!(all_d,d);push!(all_ar,best[9]);push!(all_bx,best[5])
        push!(all_curves,best[6]);push!(all_seeds_run,length(d_res))
        update_init_layer!(init_layer,d,best[5])
    end
    run_phase2_and_summary(all_d,all_ar,all_bx,all_curves,all_seeds_run,init_layer,n_bo)
    exit(0)
end

#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ── DISPATCH ────────────────────────────────────────────────────────────────
# Follows the template exactly: @time pmap(compute_task, all_params)

# ── DEFAULT Q3 SWEEP (runs only when no mode arg is given) ─────────────────
if isempty(ARGS) || ARGS[1] == "sweep"
    mkpath(directoryBasis)
    println("=-"^50)
    println("PARALLEL COMPUTING — $(JOB_LABEL)")
    println("Total tasks:     $(N_TASKS)  ($(length(D_CANDIDATES)) d-values × $(length(SEEDS)) seeds)")
    println("Workers:         $(nworkers())")
    println("Budget/task:     $(QAOA_BUDGET) BO evals")
    println("Output dir:      $(directoryBasis)")
    println("Started:         $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("=-"^50)
    flush(stdout)
    
    @time results = pmap(compute_task, all_params)
    
    #----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
    #----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
    # ── COLLECT RESULTS ─────────────────────────────────────────────────────────
    # Unpack result tuples (task_id,d,seed,by,bx,curve,nevals,nedge,ar,elapsed)
    
    println("\n=== Phase 1 results (all tasks) ===")
    println("╔"*"═"^72*"╗")
    println("║  task  │  d (μm)  │  seed  │  ar      │  cut     │  evals  │  time  ║")
    println("╠"*"═"^72*"╣")
    for r in sort(results, by=x->x[1])
        (tid,d,seed,by,bx,curve,nevals,nedge,ar,elapsed) = r
        @printf("║  %3d   │  %6.1f  │  %5d │  %.4f  │  %.4f  │  %5d  │ %5.0fs ║\n",
            tid,d,seed,ar,-by,nevals,elapsed)
    end
    println("╚"*"═"^72*"╝")
    
    # Save per-task result files (for aggregate mode / restart)
    for r in results
        (tid,d,seed,by,bx,curve,nevals,nedge,ar,elapsed) = r
        fpath = joinpath(directoryBasis,"task_$(lpad(tid,3,'0')).jls")
        serialize(fpath, r)
    end
    println("\nPer-task results saved to: $(directoryBasis)/")
    
    # Best result per d-value
    init_layer=InitLayer(); all_d=Float64[]; all_ar=Float64[]
    all_bx=Vector{Vector{Float64}}(); all_curves=Vector{Vector{Float64}}()
    all_seeds_run=Int[]
    
    println("\n=== Phase 1 best-per-d ===")
    println("╔"*"═"^70*"╗")
    println("║  d (μm)  │  best ar  │  cut value  │  seeds  │  best seed      ║")
    println("╠"*"═"^70*"╣")
    for (di,d) in enumerate(D_CANDIDATES)
        d_res=[r for r in results if r[2]==d]
        isempty(d_res) && continue
        best=argmax(r->r[9],d_res)   # argmax(f,col) returns element, not index
        (tid,_,seed,by,bx,curve,nevals,nedge,ar,_)=best
        push!(all_d,d);push!(all_ar,ar);push!(all_bx,bx)
        push!(all_curves,curve);push!(all_seeds_run,length(d_res))
        update_init_layer!(init_layer,d,bx)
        @printf("║  %6.1f    │  %.4f   │  %9.4f  │    %d/3   │  seed=%-9d ║\n",
            d,ar,-by,length(d_res),seed)
    end
    println("╚"*"═"^70*"╝")
    
    #----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
    #----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
    # ── SAVE RESULTS ────────────────────────────────────────────────────────────
    tag = "$(JOB_LABEL)-Ntasks-$(N_TASKS)"
    
    # Save full results array
    save_path = joinpath(directoryBasis,"results-$(tag).jls")
    serialize(save_path, results)
    println("✓ Full results saved → $(save_path)")
    
    #----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
    # ── PHASE 2 + FINAL SUMMARY ─────────────────────────────────────────────────
    run_phase2_and_summary(all_d,all_ar,all_bx,all_curves,all_seeds_run,init_layer,N_BO_ITER_D)
    
    println("=+"^50)
    println("=-"^50)
    println("All tasks complete.  Results in: $(directoryBasis)")
    println("Finished: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("=-"^50)
    println("=+"^50)
    #----%%%%
end  # default sweep guard
#----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%


#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
#----%%%%----%%%%----%%%%----%%%%----#----%%%%----%%%%----%%%%----%%%%----%%%%
# ══ PART II: GNN WARM-START + LARGER GRAPHS ════════════════════════════════
#
# Implements Graph Neural Network initialisation of QAOA following:
#   Jain, Coyle, Kashefi, Kumar (2022) "Graph neural network initialisation
#   of quantum approximate optimisation", Quantum 2022.
#
# Key adaptations for Rydberg hardware:
#   1. GNN output p_i feeds into HARDWARE-NATIVE warm-start:
#      Detuning schedule Delta_i^WS = 2*V_i * (2*p_i - 1), where V_i is the
#      local Rydberg interaction strength. This goes beyond standard circuit
#      warm-start (which only modifies the mixer Hamiltonian) by embedding the
#      GNN solution directly into the physical detuning waveform.
#   2. LGNN (Line Graph Neural Network, Chen et al. 2019) is used as in paper.
#   3. Unsupervised training: loss = (2p-1)^T L_G (2p-1) (QUBO MaxCut).
#   4. Warm-start chaining: GNN parameters trained on small graphs generalise
#      to Q4 (16-atom), 3-regular (16/20-atom), and unit-disk graphs.
#
# Dispatch modes added:
#   julia Q3_sge.jl gnn_train <graph_type> <n_atoms> [n_epochs]
#   julia Q3_sge.jl gnn_qaoa  <graph_type> <n_atoms> <p_val>
#   julia Q3_sge.jl larger_graph <graph_type> <n_atoms> <p_val> [d_fixed]
#
# Graph types: q3 | q4 | reg3 | udisk
# ═══════════════════════════════════════════════════════════════════════════

using LinearAlgebra, Statistics

# ── GRAPH CONSTRUCTION ───────────────────────────────────────────────────────

"""Build Q4 hypercube: 16 atoms in 4D, nearest-neighbour distance d.
   Embedded in 3D: x=(bit0*d, bit1*d, bit2*d), bit3 encoded in z-offset."""
function make_q4_atoms(d::Float64)
    positions = Tuple{Float64,Float64,Float64}[]
    for i in 0:15
        x = Float64((i     ) & 1) * d
        y = Float64((i >> 1) & 1) * d
        z = Float64((i >> 2) & 1) * d + Float64((i >> 3) & 1) * (2.5*d)  # 4th dim → z-layer
        push!(positions, (x, y, z))
    end
    return AtomList(positions)
end

@everywhere begin
"""Q4 hypercube edges: connect atoms that differ in exactly one bit."""
function make_q4_edges(n_atoms::Int=16)
    edges = Tuple{Int,Int}[]
    for i in 0:(n_atoms-1)
        for j in (i+1):(n_atoms-1)
            if count_ones(xor(i, j)) == 1  # exactly one bit different
                push!(edges, (i+1, j+1))
            end
        end
    end
    return edges
end

"""Build a random 3-regular graph on n_atoms nodes (n_atoms must be even).
   Uses configuration model: each node gets 3 stubs, stubs are paired randomly.
   Retries until a simple graph (no self-loops, no multi-edges) is obtained."""
function make_reg3_graph(n_atoms::Int, seed::Int=42)
    # [FIX] 原来用 Random.seed!(seed) + 全局 RNG。Julia 1.11 换掉了默认 RNG 的
    # 实现, 同一个种子在 1.10 和 1.11 上产生的随机序列**不同** —— 实测本机
    # (1.10.4)选出 seed=2/4 两张图, 集群(1.11.7)同样参数选出 seed=1/7, 而且
    # 质量差很多(unwanted 0.20 vs 0.11)。后果是本机和集群的结果不能混用,
    # 换 Julia 版本也无法复现, 这对论文是硬伤。
    # 改用显式的 MersenneTwister 实例: 它的算法跨 Julia 版本稳定, 而且不再
    # 依赖/污染全局 RNG 状态。
    rng = MersenneTwister(seed)
    @assert n_atoms % 2 == 0 && n_atoms * 3 % 2 == 0
    max_attempts = 200
    for _ in 1:max_attempts
        stubs = shuffle(rng, repeat(1:n_atoms, 3))   # 3 stubs per node
        edges = Tuple{Int,Int}[]
        ok = true
        for k in 1:2:length(stubs)
            u, v = stubs[k], stubs[k+1]
            u == v && (ok=false; break)           # self-loop
            (min(u,v), max(u,v)) in edges && (ok=false; break)  # multi-edge
            push!(edges, (min(u,v), max(u,v)))
        end
        ok && return edges
    end
    error("Could not generate simple 3-regular graph after $max_attempts attempts")
end

"""Place n_atoms atoms on a 2D grid for a given graph, with spacing d.
   Grid layout: ceil(sqrt(n_atoms)) × ceil(sqrt(n_atoms)) square lattice."""
function make_grid_atoms(n_atoms::Int, d::Float64)
    cols = ceil(Int, sqrt(n_atoms))
    positions = Tuple{Float64,Float64,Float64}[]
    for i in 0:(n_atoms-1)
        x = Float64(i % cols) * d
        y = Float64(i ÷ cols) * d
        push!(positions, (x, y, 0.0))
    end
    return AtomList(positions)
end

"""Build a unit-disk random graph: n_atoms placed randomly in a 2D box,
   edges connect all pairs within radius r_c = 1.5*d."""
function make_udisk_graph(n_atoms::Int, d::Float64, seed::Int=42; r_factor::Float64=1.5)
    Random.seed!(seed)
    box = d * sqrt(n_atoms) * 0.8   # scale box so density ≈ 3-regular
    positions = [(rand()*box, rand()*box, 0.0) for _ in 1:n_atoms]
    atoms = AtomList(positions)
    r_c = r_factor * d
    edges = build_edges_by_radius(atoms, r_c)
    return atoms, edges
end

"""Generic graph → atoms + edges for any supported graph type.
   Returns (atoms, edges, n_atoms, n_edges, graph_label)."""
function make_graph(graph_type::String, n_atoms::Int, d::Float64;
                    seed::Int=42)::Tuple
    if graph_type == "q3"
        atoms = make_cube_atoms(d)
        edges = build_edges_by_radius(atoms, 1.2*d)
        label = "Q3-hypercube(n=8)"
    elseif graph_type == "q4"
        # [FIX] 原来的 make_q4_atoms 用固定公式摆位置, 第4维(bit3翻转)的边
        # 距离被迫是 2.5d (否则跟其它顶点位置重合), 跟其它3个方向的 d 不一致。
        # 改成: 拓扑(哪些顶点相连)不变, 位置换成统一目标距离 d 的嵌入。
        edges = make_q4_edges(16)
        atoms, stress_val = embed_unweighted_graph_2d(edges, 16, d; seed=seed)
        stress_val > 1.0 && @warn "Q4 图嵌入残差较大 (stress=$(round(stress_val,digits=3))), 并非每条边都精确达到目标距离 d=$(d)"
        label = "Q4-hypercube(n=16)"
    elseif graph_type == "reg3"
        # [FIX] 原来的 make_grid_atoms 按网格线性摆位置, 跟 make_reg3_graph
        # 生成的随机拓扑完全无关, 边距离基本是不受控的 (d 到 3d+ 都可能)。
        # 改成: 拓扑不变, 位置换成统一目标距离 d 的嵌入。
        edges = make_reg3_graph(n_atoms, seed)
        atoms, stress_val = embed_unweighted_graph_2d(edges, n_atoms, d; seed=seed)
        stress_val > 1.0 && @warn "reg3 图嵌入残差较大 (stress=$(round(stress_val,digits=3))), 图规模较大时属于正常现象"
        label = "3-regular(n=$(n_atoms))"
    elseif graph_type == "udisk"
        # [FIX] 原来的随机散布+阈值判边, 边距离连续分布在 (0,1.5d], 不一致。
        # 拓扑生成方式不变 (还是先随机散布判定哪些点连边), 只是位置作废后
        # 重新嵌入到统一目标距离 d, 保留"随机几何图"的度数分布特征。
        _, edges = make_udisk_graph(n_atoms, d, seed)
        atoms, stress_val = embed_unweighted_graph_2d(edges, n_atoms, d; seed=seed)
        stress_val > 1.0 && @warn "udisk 图嵌入残差较大 (stress=$(round(stress_val,digits=3)))"
        label = "unit-disk(n=$(n_atoms))"
    elseif graph_type == "honeycomb"
        # [新增] 论文 Bauer et al. Method B 自己测"随机图"时用的构造方式:
        # 拓扑直接从蜂窝格点邻接关系生成, 不是独立随机拓扑套到无关格子上。
        # 边全部严格等长=d, 不需要嵌入(没有"没对上"这回事), 但degree不保证
        # 精确=3(边界效应, 会打印实际度数分布)。跟 target/force 那几种"加权
        # 随机3-regular图"方案是两条不同的路, 这个是独立的无权图类型。
        atoms, edges = make_honeycomb_graph(n_atoms, d)
        label = "honeycomb(n=$(length(atoms)))"
    else
        error("Unknown graph type: $(graph_type). Use: q3 | q4 | reg3 | udisk | honeycomb")
    end
    return atoms, edges, length(atoms), length(edges), label
end # @everywhere begin (graph construction)
end

# ── LGNN IMPLEMENTATION (Line Graph Neural Network) ──────────────────────────
# Adapted from: Chen, Li, Bruna (2019), "Supervised Community Detection with
# Line Graph Neural Networks" (ICLR 2019); as used in Jain et al. (2022).
#
# Pure Julia implementation — no Python dependencies.
# Architecture:
#   • Two parallel GNNs: one on G (node features h), one on L(G) (edge features g)
#   • Non-backtracking operator B for L(G) updates
#   • Updates alternate: h_{t+1} depends on h_t and g_t; g_{t+1} likewise
#   • Output: softmax over final h gives node probabilities p_i

struct LGNNParams
    # GNN on G: J+3 weight matrices, each (feat_dim, feat_dim)
    theta::Vector{Matrix{Float64}}   # length J+3
    # GNN on L(G): J+3 weight matrices
    phi::Vector{Matrix{Float64}}     # length J+3
    J::Int          # number of power-graph adjacency levels
    feat_dim::Int   # feature dimension d_h
    T_iter::Int     # number of LGNN iterations
    # Output layer: feat_dim → 2  [FIX: was documented as 2*feat_dim, but
    # lgnn_forward truncates h back to feat_dim every iteration (see
    # `h_new = h_new[1:d, :]`) before ever reaching the output layer, so the
    # actual input to W_out is feat_dim-dimensional, not 2*feat_dim.]
    W_out::Matrix{Float64}
    b_out::Vector{Float64}
end

function init_lgnn(J::Int=3, feat_dim::Int=32, T_iter::Int=8)
    scale = 0.1
    nm = lgnn_n_mats(J)   # 见 LGNN_SEPARATE_LINE_WEIGHTS: 默认 J+3(现有行为), 开关打开后 J+4
    mk_w() = scale * randn(feat_dim, feat_dim)
    theta = [mk_w() for _ in 1:nm]
    phi   = [mk_w() for _ in 1:nm]
    W_out = scale * randn(2, feat_dim)   # [FIX: was (2, 2*feat_dim), dimension mismatch]
    b_out = zeros(2)
    return LGNNParams(theta, phi, J, feat_dim, T_iter, W_out, b_out)
end

"""Build power adjacency matrices A_j (j=0..J) and adjacency A for graph with
   n nodes and given edge list. A_0 = I, A_1 = A, A_j = min(1, A^{2^j})."""
function build_power_adj(n::Int, edges::Vector{Tuple{Int,Int}}, J::Int)
    # Standard adjacency matrix
    A = zeros(n, n)
    for (i,j) in edges
        A[i,j] = 1.0; A[j,i] = 1.0
    end
    D = Diagonal(vec(sum(A, dims=2)))
    # Power adjacency matrices
    Ajs = Matrix{Float64}[]
    push!(Ajs, A)
    Acur = A
    for _ in 1:(J-1)
        Acur = min.(1.0, Acur * Acur)
        push!(Ajs, Acur)
    end
    return A, D, Ajs
end

"""Build line graph structures: L(G) nodes = edges of G,
   non-backtracking operator B, signed/unsigned incidence matrices S,U."""
function build_line_graph(n::Int, edges::Vector{Tuple{Int,Int}})
    m = length(edges)   # number of edges = nodes in L(G)
    edge_idx = Dict(e => k for (k,e) in enumerate(edges))
    for (k,(i,j)) in enumerate(edges)
        edge_idx[(j,i)] = k  # both orientations
    end

    # Non-backtracking operator B: B[(i→j),(k→l)] = 1 if j=k and i≠l
    # L(G) is undirected, so we use the symmetrised version
    B = zeros(m, m)
    for (e1, k) in edge_idx
        i, j = e1
        for (e2, l) in edge_idx
            k_src, l_src = e2
            if l_src == i && k_src != j    # j→k, k→l where k=j and l≠i (no backtrack)
                # Actually: B[(i→j),(k→l)] = 1 iff j=k and i≠l
                # But since edges are undirected, we symmetrise
            end
        end
    end
    # Simpler: use the symmetric L(G) adjacency (two edges share a node)
    BL = zeros(m, m)
    for k in 1:m, l in 1:m
        k == l && continue
        i1, j1 = edges[k]; i2, j2 = edges[l]
        if i1==i2 || i1==j2 || j1==i2 || j1==j2
            BL[k,l] = 1.0
        end
    end
    D_L = Diagonal(vec(sum(BL, dims=2)))

    # Unsigned incidence U: U[i, (k→l)] = 1 if i is an endpoint of edge (k→l)
    U = zeros(n, m)
    for (k,(i,j)) in enumerate(edges)
        U[i,k] = 1.0; U[j,k] = 1.0
    end
    # Signed incidence S: S[i,(k→l)] = 1 if i=k, -1 if i=l
    S = zeros(n, m)
    for (k,(i,j)) in enumerate(edges)
        S[i,k] = 1.0; S[j,k] = -1.0
    end

    return BL, D_L, U, S
end

relu(x) = max(0.0, x)
relu_mat(X) = max.(0.0, X)

# ══════════════════════════════════════════════════════════════════════════
# [重构] lgnn_forward 拆成"预计算"+"可微核心"两半 —— 目的是让 Zygote
# (反向模式自动微分) 能用在这个网络上。原来的写法有两处 Zygote 硬性不支持
# 的东西: (1) `h = zeros(d,n)` 之后逐元素 `h[1,i] = ...` 这种**原地修改
# 数组**, Zygote 会直接报 "Mutating arrays is not supported"; (2) eigen
# 特征分解。好在这两样**都不依赖网络参数**(只跟图的拓扑/权重有关), 所以
# 整体挪到 lgnn_precompute 里做一次就行, 完全不影响需要求导的那部分。
# lgnn_core 是纯函数式、无原地修改的版本, Zygote 可以直接对它求导。
# lgnn_forward 保留原来的调用签名, 内部改成"先 precompute 再 core", 所以
# 所有已有的 SPSA 调用点一行都不用改, 数学上也完全等价。
#
# [说明] mean/std 这里手写展开成 sum/sqrt 的形式(跟 Statistics.std 的
# corrected=true, 也就是 n-1 分母, 数学上完全一致), 不是为了改数值, 是因为
# 带 dims 关键字的 std 在 Zygote 里的求导规则支持情况我没法在这里实测验证,
# 手写展开是确定安全的写法。浮点层面可能有最后几位的差异, 不影响结论。
# ══════════════════════════════════════════════════════════════════════════

# [开关] 见聊天记录讨论: theta 需要承担 2(self/degree) + J(幂邻居) +
# 2(线图有符号/无符号) = J+4 个不同角色, 但 init_lgnn 只分配了 J+3 个矩阵,
# 导致 theta[2+J] 和 theta[J+2] 撞在同一个下标上——"J阶幂邻居"和"线图无符号
# 消息"这两种语义完全不同的消息被迫共用同一个权重矩阵(phi 同理)。这几乎
# 肯定是索引笔误而不是有意设计(论文里 LGNN 每一项都是独立权重)。
# 默认保持 false = 维持现有行为, 这样这次换 Zygote 优化器的效果能被干净地
# 单独归因; 想验证架构修复的增量时, 单独把这个改成 true 再跑一次即可
# (注意: 改了之后参数量会变, 之前存的 prior 缓存文件不能再用, 要重新训练)。
const LGNN_SEPARATE_LINE_WEIGHTS = false

lgnn_n_mats(J::Int) = LGNN_SEPARATE_LINE_WEIGHTS ? J+4 : J+3

# 手写的按列 mean/std 和全局 std —— 跟 Statistics 的对应函数数学等价,
# 但保证是 Zygote 友好的纯广播写法, 见上面注释。
_lg_colmean(x) = sum(x, dims=2) ./ size(x, 2)
_lg_colstd(x)  = sqrt.(sum((x .- _lg_colmean(x)).^2, dims=2) ./ (size(x, 2) - 1))
_lg_allstd(x)  = sqrt(sum((x .- sum(x)/length(x)).^2) / (length(x) - 1))

"""LGNN 前向传播里**不依赖网络参数**的那部分, 只跟图本身有关, 每张图算一次
   就能反复用。包含: 拉普拉斯特征向量位置编码(eigen)、初始节点/边特征
   h0/g0 的组装(这两步都有原地数组赋值, 放在这里做, 不进入求导路径)、
   以及线图幂邻接矩阵 Bjs(原来在 T 轮循环里每轮重算一遍, 既浪费也没必要)。"""
function lgnn_precompute(A::Matrix, D::Diagonal, Ajs::Vector{Matrix{Float64}},
                          BL::Matrix, D_L::Diagonal, U::Matrix, S::Matrix,
                          n::Int, m::Int, d::Int, J::Int;
                          node_feat1::Union{Nothing,Vector{Float64}}=nothing,
                          edge_feat1::Union{Nothing,Vector{Float64}}=nothing)
    # 拉普拉斯特征向量位置编码: 取**最大**特征值对应的特征向量(谱的顶部),
    # 不是 Fiedler 向量那一端 —— Max-Cut 要最大化边两端差异, 跟图聚类方向相反。
    L_topo = Matrix(D) - A
    eig_n  = eigen(Symmetric((L_topo .+ L_topo') ./ 2))
    pe2_n  = n >= 2 ? eig_n.vectors[:,n]   .* sqrt(n) : zeros(n)
    pe3_n  = n >= 3 ? eig_n.vectors[:,n-1] .* sqrt(n) : zeros(n)

    L_line = Matrix(D_L) - BL
    eig_m  = eigen(Symmetric((L_line .+ L_line') ./ 2))
    pe2_m  = m >= 2 ? eig_m.vectors[:,m]   .* sqrt(m) : zeros(m)
    pe3_m  = m >= 3 ? eig_m.vectors[:,m-1] .* sqrt(m) : zeros(m)

    h0 = zeros(d, n)
    for i in 1:n
        h0[1,i] = node_feat1 === nothing ? D[i,i] : node_feat1[i]
        h0[2,i] = pe2_n[i]
        d >= 3 && (h0[3,i] = pe3_n[i])
    end
    g0 = zeros(d, m)
    for k in 1:m
        g0[1,k] = edge_feat1 === nothing ? D_L[k,k] : edge_feat1[k]
        g0[2,k] = pe2_m[k]
        d >= 3 && (g0[3,k] = pe3_m[k])
    end

    Bjs = Matrix{Float64}[min.(1.0, BL^(2^(j-1))) for j in 1:J]

    return (h0=h0, g0=g0, Ajs=Ajs, Bjs=Bjs,
            Dm=Matrix(D), DLm=Matrix(D_L), U=U, S=S, n=n, m=m)
end

"""LGNN 前向传播里依赖网络参数的那部分 —— 纯函数式, 没有任何原地数组修改,
   Zygote 可以直接对它求导。参数按 theta/phi/W_out/b_out 四块分开传, 不传
   LGNNParams 结构体, 这样 Zygote 那边可以直接对一个扁平参数向量求导, 不用
   处理结构体的梯度类型。"""
function lgnn_core(theta, phi, W_out, b_out, J::Int, T::Int, d::Int, pre)
    h = pre.h0; g = pre.g0
    # 见 LGNN_SEPARATE_LINE_WEIGHTS 的说明: 默认(false)时这两个下标会跟
    # 幂邻居循环里的 theta[2+J] 撞车, 是保留的现有行为。
    i_lgu = LGNN_SEPARATE_LINE_WEIGHTS ? J+3 : J+2
    i_lgs = LGNN_SEPARATE_LINE_WEIGHTS ? J+4 : J+3

    for t in 1:T
        # ── 更新 h (原图 G 上的 GNN) ──────────────────────────────────────
        y_bar = theta[1] * h
        y_bar = y_bar + theta[2] * (pre.Dm * h')'
        for j in 1:J
            y_bar = y_bar + theta[2+j] * (pre.Ajs[j] * h')'
        end
        y_bar = y_bar + theta[i_lgu] * (pre.U * g')'
        y_bar = y_bar + theta[i_lgs] * (pre.S * g')'
        y_h = max.(0.0, y_bar)
        h_new = vcat(y_h, y_bar)[1:d, :]
        # 每轮做均值中心化+尺度归一化: 否则 T 轮线性变换+ReLU 叠加会让共模
        # 信号指数增长, 到输出层把 softmax 推到饱和, 逐节点差异被完全冲没。
        h_new = h_new .- _lg_colmean(h_new)
        h_new = h_new ./ (_lg_colstd(h_new) .+ 1e-6)

        # ── 更新 g (线图 L(G) 上的 GNN) ───────────────────────────────────
        z_bar = phi[1] * g
        z_bar = z_bar + phi[2] * (pre.DLm * g')'
        for j in 1:J
            z_bar = z_bar + phi[2+j] * (pre.Bjs[j] * g')'
        end
        z_bar = z_bar + phi[i_lgu] * (pre.U' * h')'
        z_bar = z_bar + phi[i_lgs] * (pre.S' * h')'
        y_g = max.(0.0, z_bar)
        g_new = vcat(y_g, z_bar)[1:d, :]
        g_new = g_new .- _lg_colmean(g_new)
        g_new = g_new ./ (_lg_colstd(g_new) .+ 1e-6)

        h = h_new; g = g_new
    end

    # 输出层: 线性 + 按整体尺度归一化 + softmax
    out = W_out * h .+ b_out
    out = out ./ (_lg_allstd(out) + 1e-6)
    out = clamp.(out, -30.0, 30.0)
    exp_out = exp.(out .- maximum(out, dims=1))
    p = exp_out[1,:] ./ vec(sum(exp_out, dims=1))
    p = clamp.(p, 1e-6, 1.0 - 1e-6)
    return p
end

"""LGNN 前向传播 (对外接口保持不变) —— 内部改成"先 precompute 再 core"。
   返回每个节点的概率 p ∈ [0,1]^n。"""
function lgnn_forward(params::LGNNParams,
                      A::Matrix, D::Diagonal, Ajs::Vector{Matrix{Float64}},
                      BL::Matrix, D_L::Diagonal, U::Matrix, S::Matrix,
                      n::Int, m::Int;
                      node_feat1::Union{Nothing,Vector{Float64}}=nothing,
                      edge_feat1::Union{Nothing,Vector{Float64}}=nothing)
    pre = lgnn_precompute(A, D, Ajs, BL, D_L, U, S, n, m,
                          params.feat_dim, params.J;
                          node_feat1=node_feat1, edge_feat1=edge_feat1)
    return lgnn_core(params.theta, params.phi, params.W_out, params.b_out,
                     params.J, params.T_iter, params.feat_dim, pre)
end

"""把 LGNNParams 摊平成一个一维向量 —— Zygote 直接对这个扁平向量求导,
   比对结构体求导省事、也少一类类型坑。"""
function lgnn_flatten(p::LGNNParams)
    return vcat(vcat([vec(M) for M in p.theta]...),
                vcat([vec(M) for M in p.phi]...),
                vec(p.W_out), p.b_out)
end

"""lgnn_flatten 的逆操作, 但直接返回 (theta, phi, W_out, b_out) 四块,
   不组装成 LGNNParams —— 这个函数会在 Zygote 的求导路径里被调用, 返回
   reshape 出来的视图即可, 不需要真的构造结构体。"""
function lgnn_unflatten_parts(v::AbstractVector, J::Int, d::Int)
    nm = lgnn_n_mats(J); sz = d*d
    theta = [reshape(v[(k-1)*sz+1 : k*sz], d, d) for k in 1:nm]
    off = nm*sz
    phi = [reshape(v[off+(k-1)*sz+1 : off+k*sz], d, d) for k in 1:nm]
    off += nm*sz
    W_out = reshape(v[off+1 : off+2d], 2, d)
    off += 2d
    b_out = v[off+1 : off+2]
    return theta, phi, W_out, b_out
end

"""扁平向量 → LGNNParams, 用于训练结束后存盘 (存下来的格式跟 SPSA 那条
   路径完全一致, 推理/warmstart 那边一行都不用改)。"""
function lgnn_from_flat(v::AbstractVector, J::Int, d::Int, T::Int)
    theta, phi, W_out, b_out = lgnn_unflatten_parts(v, J, d)
    return LGNNParams([Matrix(M) for M in theta], [Matrix(M) for M in phi],
                      J, d, T, Matrix(W_out), collect(b_out))
end

"""Unsupervised GNN training loss (QUBO MaxCut objective, eq. 15-16 in paper).
   L_G is the graph Laplacian, p is the GNN output probability vector."""
function gnn_loss(p::Vector{Float64}, L_G::Matrix{Float64})
    q = 2.0 .* p .- 1.0   # map [0,1] → [-1,1]
    return -0.25 * dot(q, L_G * q)   # = -(1/4)(2p-1)^T L_G (2p-1)
end

"""对 LGNNParams 的**全部**参数块 (theta[1..J+3]、phi[1..J+3]、W_out、b_out)
   做一次真正的"同时扰动"SPSA梯度估计+更新。
   [FIX] 单方向SPSA在~12000维参数空间下噪声太大, 实测长时间训练反而会让ar
   持续下降(不是收敛, 是随机漂移)。改成"多方向平均SPSA": 每一步抽
   n_directions 个独立随机方向, 各自估计一次梯度, 取平均 —— 计算量从2次
   前向变成 2*n_directions 次(仍然是便宜的矩阵运算, 不涉及量子模拟), 但
   梯度方向的可信度显著提高。dk=扰动幅度, ak=学习率, eval_fn(p::LGNNParams)
   ->Float64 是要最小化的loss函数。"""
function spsa_update_all(params::LGNNParams, dk::Float64, ak::Float64,
                          eval_fn::Function; n_directions::Int=5)
    theta_grad = [zeros(size(t))  for t  in params.theta]
    phi_grad   = [zeros(size(ph)) for ph in params.phi]
    Wout_grad  = zeros(size(params.W_out))
    bout_grad  = zeros(size(params.b_out))

    for _ in 1:n_directions
        d_theta = [randn(size(t)) .* dk for t in params.theta]
        d_phi   = [randn(size(ph)) .* dk for ph in params.phi]
        d_Wout  = randn(size(params.W_out)) .* dk
        d_bout  = randn(size(params.b_out)) .* dk

        theta_plus  = [params.theta[i] .+ d_theta[i] for i in eachindex(params.theta)]
        theta_minus = [params.theta[i] .- d_theta[i] for i in eachindex(params.theta)]
        phi_plus    = [params.phi[i]   .+ d_phi[i]   for i in eachindex(params.phi)]
        phi_minus   = [params.phi[i]   .- d_phi[i]   for i in eachindex(params.phi)]
        Wout_plus   = params.W_out .+ d_Wout
        Wout_minus  = params.W_out .- d_Wout
        bout_plus   = params.b_out .+ d_bout
        bout_minus  = params.b_out .- d_bout

        p_plus  = LGNNParams(theta_plus,  phi_plus,  params.J, params.feat_dim,
                              params.T_iter, Wout_plus,  bout_plus)
        p_minus = LGNNParams(theta_minus, phi_minus, params.J, params.feat_dim,
                              params.T_iter, Wout_minus, bout_minus)

        diff = eval_fn(p_plus) - eval_fn(p_minus)

        for i in eachindex(params.theta)
            theta_grad[i] .+= (diff ./ (2.0 .* d_theta[i])) ./ n_directions
        end
        for i in eachindex(params.phi)
            phi_grad[i] .+= (diff ./ (2.0 .* d_phi[i])) ./ n_directions
        end
        Wout_grad .+= (diff ./ (2.0 .* d_Wout)) ./ n_directions
        bout_grad .+= (diff ./ (2.0 .* d_bout)) ./ n_directions
    end

    new_theta = [params.theta[i] .- ak .* theta_grad[i] for i in eachindex(params.theta)]
    new_phi   = [params.phi[i]   .- ak .* phi_grad[i]   for i in eachindex(params.phi)]
    new_Wout  = params.W_out .- ak .* Wout_grad
    new_bout  = params.b_out .- ak .* bout_grad

    return LGNNParams(new_theta, new_phi, params.J, params.feat_dim,
                       params.T_iter, new_Wout, new_bout)
end

"""Compute graph Laplacian L_G = D - A."""
function graph_laplacian(n::Int, edges::Vector{Tuple{Int,Int}})
    A = zeros(n, n)
    for (i,j) in edges; A[i,j]=1.0; A[j,i]=1.0; end
    D = Diagonal(vec(sum(A, dims=2)))
    return Matrix(D - A)
end

"""带权版本的图拉普拉斯矩阵: L_G = D_w - A_w, 其中 A_w[i,j]=weights[edge (i,j)]
   (而不是硬编码的 1.0)。gnn_loss(p, L_G) = -(1/4)(2p-1)ᵀ L_G (2p-1) 这个公式本身
   不用改——只要把加权的 L_G 传进去, 它自动就是加权 MaxCut 的 QUBO 目标了,
   因为 (1/4)Σ_{(i,j)∈E} w_ij (1-z_iz_j) 展开后正好对应加权拉普拉斯的二次型。"""
function weighted_graph_laplacian(n::Int, edges::Vector{Tuple{Int,Int}}, weights::Vector{Float64})
    A = zeros(n, n)
    for (idx,(i,j)) in enumerate(edges)
        A[i,j] = weights[idx]; A[j,i] = weights[idx]
    end
    D = Diagonal(vec(sum(A, dims=2)))
    return Matrix(D - A)
end

"""每个节点的加权度数: Σ_{j:(i,j)∈E} w_ij。即便拓扑是regular(比如reg3w的3-regular
   拓扑), 由于每条边权重随机, 不同节点的加权度数天然就会不一样 —— 是一个比
   人工位置编码更贴合"加权图"这个问题本身的对称性打破信号。"""
function weighted_node_degree(n_atoms::Int, edges::Vector{Tuple{Int,Int}}, weights::Vector{Float64})
    wd = zeros(n_atoms)
    for (idx,(i,j)) in enumerate(edges)
        wd[i] += weights[idx]; wd[j] += weights[idx]
    end
    return wd
end

"""train_lgnn_quick 的带权版本: 唯一区别是用 weighted_graph_laplacian 而不是
   graph_laplacian, 以及 ar 的归一化用总权重 sum(weights) 而不是边数 length(edges)
   (无权版本里 ar=cut数/边数 是一个 [0,1] 的比例; 加权版本里 ar=cut权重/总权重
   才是同样意义下的比例, 除以边数在加权情形下没有意义)。
   [FIX] 原来 best_p/best_ar 追踪的是连续松弛值 —— 已经证实这个值会严重
   低估真实离散cut (差3倍以上都可能), 跟 gnn_train/gnn_train_weighted 撞的
   是同一个坑。现在改成追踪并返回*真实离散cut*最优的那个p, 这样下游
   (gnn_qaoa_weighted/gdqc_validate_weighted) 用到的 best_p_gnn/gnn_ar
   才是有意义的数字。epoch数也从200提到300, 跟已验证过收敛情况的
   gnn_train_weighted保持一致。"""
function train_lgnn_quick_weighted(edges::Vector{Tuple{Int,Int}}, weights::Vector{Float64},
                                    n_atoms::Int; n_epochs::Int=300, init_seed::Int=1)
    A_lg, D_lg, Ajs_lg = build_power_adj(n_atoms, edges, 3)
    BL_lg, DL_lg, U_lg, S_lg = build_line_graph(n_atoms, edges)
    L_G = weighted_graph_laplacian(n_atoms, edges, weights)
    total_w = sum(weights)
    node_wd = weighted_node_degree(n_atoms, edges, weights)   # [新增] 加权度数
    # [FIX] init_lgnn 内部用的是裸 randn(), 没有独立设种子 —— 之前会意外
    # 继承"图构造消耗剩下的全局随机数流状态", 导致换图的种子(seed)顺带把
    # GNN参数初始化的起点也跟着改了, 两个因素(图本身难度 vs GNN起点好坏)
    # 混在一起分不清。这里显式给GNN初始化单独设一个种子, 跟图构造的seed
    # 参数彻底解耦 —— 默认init_seed=1(固定值), 除非调用方主动指定不同值,
    # 换图不会再意外改变GNN的初始参数分布。
    Random.seed!(init_seed + 20000)
    params = init_lgnn(3, 32, 8)
    best_ar = 0.0
    best_ar_discrete = 0.0; best_p_discrete = fill(0.5, n_atoms)
    discrete_cut_w(pvec) = begin
        cutw = 0.0
        for (idx,(u,v)) in enumerate(edges)
            (pvec[u] > 0.5) != (pvec[v] > 0.5) && (cutw += weights[idx])
        end
        cutw / total_w
    end
    eval_fn(p) = gnn_loss(
        lgnn_forward(p, A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg, n_atoms, length(edges);
                     node_feat1=node_wd, edge_feat1=weights), L_G)
    for epoch in 1:n_epochs
        p_out = lgnn_forward(params, A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg,
                              n_atoms, length(edges); node_feat1=node_wd, edge_feat1=weights)
        ar_e = -gnn_loss(p_out, L_G) / total_w
        ar_e > best_ar && (best_ar = ar_e)
        dar = discrete_cut_w(p_out)   # [新增] 真实离散cut
        dar > best_ar_discrete && (best_ar_discrete = dar; best_p_discrete = copy(p_out))
        dk = 0.05/(epoch^0.101)
        ak = 0.005/(epoch^0.602)
        params = spsa_update_all(params, dk, ak, eval_fn)   # [FIX] 同时扰动全部参数块
    end
    return params, best_p_discrete, best_ar_discrete
end

"""生成一批加权随机3-regular图, 混合不同 n_atoms 和不同 seed —— 论文
   (Jain et al. 2022) Section 4.1 明确说训练集图不需要跟目标图一样大, 可以
   混不同规模; Table 1 直接验证了跨规模泛化。
   [注意] 预训练只需要 edges/weights/n_atoms 算图拉普拉斯矩阵, 完全不需要
   原子物理位置(GNN训练本身是纯拓扑/纯经典的, 跟 force-legacy/force3d 这些
   物理构造方案是两件正交的事, 见聊天记录讨论)——所以这里用最简单快速的
   独立随机权重(跟 make_reg3_weighted_graph 一样), 不用跑耗时的力场/嵌入
   优化, 训练池可以很快建好。"""
function build_graph_pool_weighted(n_list::Vector{Int}, seeds_per_size::Int;
                                    wmin::Float64=0.5, wmax::Float64=1.0)
    pool = NamedTuple[]
    for n_at in n_list
        for s in 1:seeds_per_size
            seed = n_at*1000 + s   # 每个(n,s)组合对应一个确定性的、彼此不同的种子
            edges = make_reg3_graph(n_at, seed)
            Random.seed!(seed + 1000)
            weights = wmin .+ rand(length(edges)) .* (wmax - wmin)
            push!(pool, (edges=edges, weights=weights, n_atoms=n_at))
        end
    end
    return pool
end

@everywhere function build_one_pool_graph_target3d(n_at::Int, seed::Int, wmin::Float64, wmax::Float64)
    atoms, edges, weights, stress_val, unwanted_ratio, fidelity =
        make_reg3_weighted_graph_target3d(n_at, seed; wmin=wmin, wmax=wmax)
    # [新增] "干净"判定: 非边最近距离是否仍然大于边的最远目标距离, 也就是
    # 边/非边两组距离分布有没有真正分开、没有重叠。之前只能靠肉眼扫日志判断
    # "这个规模行不行", 4个worker交错打印的log很容易看漏/看错(实测就发生
    # 过n=36被误判成"还凑合"), 现在直接算出来, 不用再猜。
    pos = [atoms[i] for i in 1:n_at]
    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    edge_dists = Float64[]; nonedge_dists = Float64[]
    for i in 1:n_at, j in (i+1):n_at
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        d = sqrt((xi-xj)^2+(yi-yj)^2+(zi-zj)^2)
        (i,j) in edge_set ? push!(edge_dists, d) : push!(nonedge_dists, d)
    end
    is_clean = minimum(nonedge_dists) > maximum(edge_dists)
    return (edges=edges, weights=weights, n_atoms=n_at, is_clean=is_clean)
end

"""build_graph_pool_weighted 的 target3d 版本 —— target3d 是三轮实测下来
   综合表现最好的构造方案(权重比值接近最初设定的目标、unwanted_ratio跟
   force3d-legacy一个量级、目标权重相关系数≈0.99), 定为主力方案后训练池
   也应该跟它同源构造, 不是用 force3d 或者独立随机权重凑合。
   跟原来的 build_graph_pool_weighted 比, 关键区别:
   1. 权重不再是独立随机生成后就直接用, 而是先独立生成一组目标权重(逻辑上
      跟原来一样, wmin~wmax), 再用 target3d 的三维逐边目标力场把它物理实现
      出来, 最终权重是收敛后的*实际*值——train和test用的是同一套构造管线,
      不会出现"训练池权重分布和测试图权重分布对不上"这个问题。
   2. 按单张图逐个取舍(is_clean=非边最近距离是否真的大于边最远目标距离),
      不是按规模档整体取舍。
   [FIX v3] 种子数不再是固定值——实测发现不同规模的干净比例差异巨大
   (n=12约45%, n=24约20%, n=36约0%), 固定种子数会导致: 干净比例低的规模
   在最终训练池里样本量偏少(不均衡), 干净比例为0的规模(比如n=36)不管
   加多少种子都凑不出一张能用的图(纯粹浪费算力)。改成两阶段自适应:
     第一阶段"探路": 每个规模先跑 probe_seeds 个种子, 估计这个规模的真实
       干净比例。干净比例低于 min_clean_rate(默认5%)的规模直接跳过, 不
       再往里砸种子——这不是"运气不够", 是这个规模目前超出target3d的
       嵌入能力上限, 加种子解决不了。
     第二阶段"补种": 对干净比例够高的规模, 按探路阶段估计的比例反推还
       需要跑多少个种子, 才能凑够 target_clean_per_size 张干净图, 只在
       这个数量的基础上补种, 不是所有规模都用同一个种子数。
   这样不用每次换 n_atoms 都手动试探"该给几个种子", 代码自己会根据实际
   干净比例算出来。"""
function build_graph_pool_weighted_target3d(n_test::Int, target_clean_per_size::Int;
                                             wmin::Float64=0.5, wmax::Float64=1.0,
                                             n_list::Union{Nothing,Vector{Int}}=nothing,
                                             probe_seeds::Int=15,
                                             min_clean_rate::Float64=0.05,
                                             max_fill_seeds::Int=500)
    # [FIX] 原来只覆盖 1x/2x 两个离散的规模档, 现在改成覆盖
    # [n_test, 2*n_test] 之间**所有偶数** n (n_test=12 → 12,14,16,18,20,22,24,
    # 7 个规模而不是 2 个), 训练池的规模多样性明显提升。make_reg3_graph 要求
    # n_atoms 是偶数(3-regular 图的握手定理), 所以只取偶数。
    n_start = n_test + (isodd(n_test) ? 1 : 0)
    n_list = n_list === nothing ? collect(n_start:2:(2*n_test)) : sort(unique(n_list))

    kept = NamedTuple[]
    println("  自适应种子分配 (目标: 每个规模凑够 $(target_clean_per_size) 张干净图)")
    println("  覆盖规模: $(n_list)")
    for n_at in n_list
        println("  ── n=$(n_at): 第一阶段探路, 先跑 $(probe_seeds) 个种子 ──")
        probe_tasks = [(n_at, n_at*1000+s) for s in 1:probe_seeds]
        probe_raw = pmap(t -> build_one_pool_graph_target3d(t[1], t[2], wmin, wmax), probe_tasks)
        n_clean_probe = count(r -> r.is_clean, probe_raw)
        clean_rate = n_clean_probe / probe_seeds
        @printf("    探路结果: %d/%d 干净 (%.0f%%)\n", n_clean_probe, probe_seeds, 100*clean_rate)

        if clean_rate < min_clean_rate
            @printf("    干净比例低于阈值(%.0f%%), 跳过这个规模, 不再补种子——\n", 100*min_clean_rate)
            println("    这不是运气问题, 是target3d目前的嵌入能力在n=$(n_at)上有真实上限。")
            continue
        end

        append!(kept, NamedTuple[(edges=r.edges, weights=r.weights, n_atoms=r.n_atoms) for r in probe_raw if r.is_clean])
        n_have = n_clean_probe
        if n_have < target_clean_per_size
            n_needed = min(ceil(Int, (target_clean_per_size - n_have) / clean_rate), max_fill_seeds)
            println("    第二阶段补种: 按当前比例估计, 再跑 $(n_needed) 个种子去凑够目标数量")
            fill_tasks = [(n_at, n_at*1000+probe_seeds+s) for s in 1:n_needed]
            fill_raw = pmap(t -> build_one_pool_graph_target3d(t[1], t[2], wmin, wmax), fill_tasks)
            n_clean_fill = count(r -> r.is_clean, fill_raw)
            append!(kept, NamedTuple[(edges=r.edges, weights=r.weights, n_atoms=r.n_atoms) for r in fill_raw if r.is_clean])
            @printf("    补种后 n=%3d 共 %d/%d 干净\n", n_at, n_have+n_clean_fill, probe_seeds+n_needed)
        else
            println("    探路阶段已经凑够目标数量, 不需要补种")
        end
    end

    println("  最终训练池共 $(length(kept)) 张图 (规模范围 $(isempty(kept) ? "N/A" : "$(minimum(g.n_atoms for g in kept))~$(maximum(g.n_atoms for g in kept))"))")
    if isempty(kept)
        error("所有规模的干净比例都低于阈值, 训练池为空——" *
              "n_test=$(n_test) 这个规模本身可能已经超出 target3d 目前能稳定处理的范围, " *
              "考虑换更小的 n_atoms 或者先去查 embed_reg3_force_3d_target 的收敛质量。")
    end
    return kept
end

"""GNN 预训练(加权版)—— 对应论文 Eq.15: 代价函数是训练集里 T 张图各自
   Eq.16 代价的**平均值**, 不是像 train_lgnn_quick_weighted 那样只看一张图。
   训练出来的参数不跟任何具体图绑定, 是一个可以直接用在*任意*新图上做
   一次性推理的"先验"(prior)。
   [FIX] 每个epoch如果真的对全部T张图都算一遍loss再平均, 训练会很慢——
   改成mini-batch: 每个epoch从训练池里随机抽 batch_size 张图算平均loss,
   标准的mini-batch做法, 期望值上等价于Eq.15的全批平均, 大幅降低每个epoch
   的计算量。默认 batch_size 从8提到24 —— 实测8的时候, 训练全程在随机基线
   (0.5)附近来回震荡、不收敛, 大概率是"SPSA本身的扰动噪声"和"每步抽到
   哪几张图的采样噪声"叠加导致梯度方向太不可靠, 加大batch_size能压低后者。
   [FIX] 原来只返回训练结束那一刻的参数, 但训练过程本身在震荡, 最后一轮
   很可能比中间某个检查点还差(实测就出现过这种情况)。现在跟单图训练一样,
   每次算"训练池平均真实离散cut"(不是连续松弛, 这个已经确认过会严重低估
   真实质量)时, 顺便记录是否刷新了历史最优, 训练结束后返回*历史最优*那组
   参数, 不是训练结束时的参数。"""
function train_lgnn_pretrain_weighted(pool::Vector{NamedTuple};
                                       n_epochs::Int=1000, batch_size::Int=24,
                                       init_seed::Int=1, report_every::Int=100)
    Random.seed!(init_seed + 20000)
    params = init_lgnn(3, 32, 8)

    # 预先给每张图算好各自的图矩阵, 不用每个epoch重算
    graph_data = NamedTuple[]
    for g in pool
        A_lg, D_lg, Ajs_lg = build_power_adj(g.n_atoms, g.edges, 3)
        BL_lg, DL_lg, U_lg, S_lg = build_line_graph(g.n_atoms, g.edges)
        L_G = weighted_graph_laplacian(g.n_atoms, g.edges, g.weights)
        total_w = sum(g.weights)
        node_wd = weighted_node_degree(g.n_atoms, g.edges, g.weights)
        push!(graph_data, (A_lg=A_lg, D_lg=D_lg, Ajs_lg=Ajs_lg, BL_lg=BL_lg, DL_lg=DL_lg,
                           U_lg=U_lg, S_lg=S_lg, L_G=L_G, total_w=total_w, node_wd=node_wd,
                           weights=g.weights, edges=g.edges, n_atoms=g.n_atoms))
    end

    discrete_cut_w(pvec, edges, weights, total_w) = begin
        cutw = 0.0
        for (idx,(u,v)) in enumerate(edges)
            (pvec[u] > 0.5) != (pvec[v] > 0.5) && (cutw += weights[idx])
        end
        cutw / total_w
    end

    n_pool = length(graph_data)
    eval_fn(p) = begin
        idxs = rand(1:n_pool, min(batch_size, n_pool))
        total = 0.0
        for idx in idxs
            gd = graph_data[idx]
            pout = lgnn_forward(p, gd.A_lg, gd.D_lg, gd.Ajs_lg, gd.BL_lg, gd.DL_lg, gd.U_lg, gd.S_lg,
                                 gd.n_atoms, length(gd.edges); node_feat1=gd.node_wd, edge_feat1=gd.weights)
            total += gnn_loss(pout, gd.L_G)
        end
        total / length(idxs)
    end

    local best_params, best_avg
    best_params = deepcopy(params); best_avg = 0.0

    for epoch in 1:n_epochs
        # [试验性调整] 扰动幅度衰减指数从标准SPSA推荐的0.101提到0.2, 让扰动
        # 随训练推进收窄得更明显——这不是教科书标准做法(标准做法里dk本该比
        # ak衰减得慢, 避免扰动太小时有限差分噪声被放大), 但实测batch_size=24
        # 那次训练在epoch1000找到高点(0.5914)后被甩出去、之后900轮都没追回来,
        # 这个"找到好区域又被踢出去"的症状是扰动幅度全程没怎么收窄的典型表现。
        # 只在预训练这个函数里试, 不动单图训练那几个函数(那边没有出现同样症状)。
        dk = 0.05/(epoch^0.2)
        ak = 0.005/(epoch^0.602)
        params = spsa_update_all(params, dk, ak, eval_fn)

        if epoch % report_every == 0
            total_discrete = 0.0
            for gd in graph_data
                pout = lgnn_forward(params, gd.A_lg, gd.D_lg, gd.Ajs_lg, gd.BL_lg, gd.DL_lg, gd.U_lg, gd.S_lg,
                                     gd.n_atoms, length(gd.edges); node_feat1=gd.node_wd, edge_feat1=gd.weights)
                total_discrete += discrete_cut_w(pout, gd.edges, gd.weights, gd.total_w)
            end
            avg_discrete = total_discrete / n_pool
            is_best = avg_discrete > best_avg
            is_best && (best_avg = avg_discrete; best_params = deepcopy(params))
            @printf("  epoch %4d  训练池平均真实离散cut = %.4f  best=%.4f%s  (跨 %d 张图, n_atoms范围 %d~%d)\n",
                    epoch, avg_discrete, best_avg, is_best ? " ★新高" : "", n_pool,
                    minimum(g.n_atoms for g in graph_data), maximum(g.n_atoms for g in graph_data))
        end
    end
    return best_params, best_avg
end

# ══════════════════════════════════════════════════════════════════════════
# ── Zygote (反向模式自动微分) 版预训练 ─────────────────────────────────────
# 为什么值得换掉 SPSA: init_lgnn(3,32,8) 的参数量是
#   theta 6×(32×32) + phi 6×(32×32) + W_out 2×32 + b_out 2 ≈ 12,354 个。
# SPSA 每一步只做 2 次前向评估, 用一个随机方向的差分去估计整个梯度 —— 不管
# 参数是 12 个还是 12,354 个, 每步拿到的信息量都一样, 本质上是把 12,354 维
# 的真实梯度投影到 1 个(加了 n_directions 也就 5 个)随机方向上, 信噪比大约
# 只有 √(5/12354) ≈ 2%。这很可能就是我们前面几轮反复调 batch_size / 扰动
# 衰减 / 种子分配, 训练曲线却始终在随机基线(0.5)附近晃的真正原因 —— 问题
# 不在训练池怎么构造, 在优化器本身撑不起这个参数规模。
# 反向模式 AD 一次反传就能拿到**全部 12,354 个参数的精确梯度**, 代价约等于
# 2~3 次前向传播, **与参数量无关** —— 每个 epoch 的开销差不多, 但拿到的是
# 精确梯度而不是 2% 信噪比的估计。
#
# [未经实测] 这段代码是照 Zygote 的要求仔细写的, 但我没有能运行 Julia 的
# 环境, 没法在交付前实跑验证。第一次跑起来如果报错(最可能是某个算子的
# 求导规则没覆盖到), 把报错发我, 一般改一两处写法就能过。
# 需要先装包: 在 ChenQAOA 环境里 `] add Zygote`。
# ══════════════════════════════════════════════════════════════════════════

"""确保 Zygote 可用: 先试着直接 using; 没装的话自动 Pkg.add 一次再 using。
   返回 true/false, 让调用方能在装不上的时候干净地退回 SPSA, 而不是直接崩。
   [集群注意] 很多集群的计算节点没有外网, Pkg.add 会失败并退回 SPSA ——
   要用 Zygote 的话建议在登录节点先跑一次(或直接 `] add Zygote`)把包装好,
   计算节点只要能读到同一个 ChenQAOA 环境就能直接 using 上。
   [注意] Pkg.add 会写入 ChenQAOA 的 Project.toml / Manifest.toml。
   只有主进程需要 Zygote(预训练全程在主进程跑, worker 只负责建图和 QAOA
   角度搜索), 所以不用 @everywhere, 也不会有多个 worker 抢着装包的问题。"""
function ensure_zygote()
    try
        @eval using Zygote
        return true
    catch
        println("  未检测到 Zygote, 正在自动安装(只需一次, 之后会缓存)...")
        try
            @eval using Pkg
            @eval Pkg.add("Zygote")
            @eval using Zygote
            println("  ✓ Zygote 安装完成")
            return true
        catch e
            @warn "Zygote 自动安装失败(集群计算节点常见: 没有外网)" exception=e
            return false
        end
    end
end

"""一个 mini-batch 的平均 Eq.16 代价, 写成"扁平参数向量 → 标量"的形式,
   Zygote 直接对这个函数求导。graph_data 里每张图的 pre 已经预计算好了
   (lgnn_precompute), 不进入求导路径。"""
function lgnn_batch_loss(v::AbstractVector, graph_data, idxs, J::Int, T::Int, d::Int)
    theta, phi, W_out, b_out = lgnn_unflatten_parts(v, J, d)
    total = 0.0
    for idx in idxs
        gd = graph_data[idx]
        p = lgnn_core(theta, phi, W_out, b_out, J, T, d, gd.pre)
        q = 2.0 .* p .- 1.0
        total += -0.25 * dot(q, gd.L_G * q)
    end
    return total / length(idxs)
end

"""train_lgnn_pretrain_weighted 的 Zygote + Adam 版本。
   跟 SPSA 版的区别只有"怎么算梯度"这一步, 代价函数(论文 Eq.15 的
   batch 平均)、监控指标(训练池平均真实离散cut)、保留历史最优这些完全一致,
   方便直接对比两者。
   早停: 先至少跑满 min_epochs(默认2000, 这是你定的最低标准), 之后如果连续
   patience 次检查都没刷新历史最优就停 —— 既省时间, 也避免我们之前看到的
   "跑到后期反而被震荡带偏"。"""
function train_lgnn_pretrain_weighted_zygote(pool::Vector{NamedTuple};
                                              n_epochs::Int=8000, min_epochs::Int=2000,
                                              batch_size::Int=24, lr::Float64=1e-3,
                                              init_seed::Int=1, report_every::Int=100,
                                              patience::Int=8)
    J = 3; d = 32; T = 8
    Random.seed!(init_seed + 20000)
    v = lgnn_flatten(init_lgnn(J, d, T))
    nparam = length(v)

    # 每张图的图矩阵 + lgnn_precompute 都只算一次, 之后每个 epoch 直接复用
    graph_data = NamedTuple[]
    for gph in pool
        A_lg, D_lg, Ajs_lg = build_power_adj(gph.n_atoms, gph.edges, J)
        BL_lg, DL_lg, U_lg, S_lg = build_line_graph(gph.n_atoms, gph.edges)
        L_G = weighted_graph_laplacian(gph.n_atoms, gph.edges, gph.weights)
        node_wd = weighted_node_degree(gph.n_atoms, gph.edges, gph.weights)
        pre = lgnn_precompute(A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg,
                              gph.n_atoms, length(gph.edges), d, J;
                              node_feat1=node_wd, edge_feat1=gph.weights)
        push!(graph_data, (pre=pre, L_G=L_G, edges=gph.edges, weights=gph.weights,
                           total_w=sum(gph.weights), n_atoms=gph.n_atoms))
    end
    n_pool = length(graph_data)

    discrete_cut_w(pvec, edges, weights, total_w) = begin
        cutw = 0.0
        for (i,(u,vv)) in enumerate(edges)
            (pvec[u] > 0.5) != (pvec[vv] > 0.5) && (cutw += weights[i])
        end
        cutw / total_w
    end

    # Adam (自己写, 不引入 Flux 依赖; 这部分在求导路径之外, 可以原地修改)
    mt = zeros(nparam); vt = zeros(nparam)
    β1, β2, ϵ = 0.9, 0.999, 1e-8

    local best_v, best_avg, since_best
    best_v = copy(v); best_avg = 0.0; since_best = 0

    @printf("  Zygote+Adam 预训练: 参数量=%d, 训练池=%d张图, lr=%.1e, batch=%d\n",
            nparam, n_pool, lr, batch_size)
    @printf("  最少跑 %d 轮, 之后连续 %d 次检查没进步就早停 (最多 %d 轮)\n",
            min_epochs, patience, n_epochs)

    for epoch in 1:n_epochs
        idxs = rand(1:n_pool, min(batch_size, n_pool))
        gradv = first(Zygote.gradient(vv -> lgnn_batch_loss(vv, graph_data, idxs, J, T, d), v))
        gradv === nothing && (gradv = zeros(nparam))

        @. mt = β1*mt + (1-β1)*gradv
        @. vt = β2*vt + (1-β2)*gradv^2
        mhat = mt ./ (1 - β1^epoch)
        vhat = vt ./ (1 - β2^epoch)
        @. v -= lr * mhat / (sqrt(vhat) + ϵ)

        if epoch % report_every == 0
            theta, phi, W_out, b_out = lgnn_unflatten_parts(v, J, d)
            total_discrete = 0.0
            for gd in graph_data
                pout = lgnn_core(theta, phi, W_out, b_out, J, T, d, gd.pre)
                total_discrete += discrete_cut_w(pout, gd.edges, gd.weights, gd.total_w)
            end
            avg_discrete = total_discrete / n_pool
            is_best = avg_discrete > best_avg
            if is_best
                best_avg = avg_discrete; best_v = copy(v); since_best = 0
            else
                since_best += 1
            end
            @printf("  epoch %5d  训练池平均真实离散cut = %.4f  best=%.4f%s  (跨 %d 张图)\n",
                    epoch, avg_discrete, best_avg, is_best ? " ★新高" : "", n_pool)
            if epoch >= min_epochs && since_best >= patience
                @printf("  早停: 已跑满最低 %d 轮, 且连续 %d 次检查没有刷新历史最优\n",
                        min_epochs, patience)
                break
            end
        end
    end

    return lgnn_from_flat(best_v, J, d, T), best_avg
end

"""用一份已经预训练好的先验参数, 对*任意*新图做一次前向推理 —— 不训练,
   不需要针对这张图重新优化。对应论文 (Jain et al. 2022) 的描述: "训练完
   GNN之后, 可以直接用这组编码来处理当前问题"。返回 (p_i向量, 真实离散cut)。"""
function gnn_infer_weighted(params::LGNNParams, edges::Vector{Tuple{Int,Int}},
                             weights::Vector{Float64}, n_atoms::Int)
    A_lg, D_lg, Ajs_lg = build_power_adj(n_atoms, edges, 3)
    BL_lg, DL_lg, U_lg, S_lg = build_line_graph(n_atoms, edges)
    node_wd = weighted_node_degree(n_atoms, edges, weights)
    p = lgnn_forward(params, A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg, n_atoms, length(edges);
                      node_feat1=node_wd, edge_feat1=weights)
    total_w = sum(weights)
    cutw = 0.0
    for (idx,(u,v)) in enumerate(edges)
        (p[u] > 0.5) != (p[v] > 0.5) && (cutw += weights[idx])
    end
    return p, cutw/total_w
end


"""Approximate gradient of GNN loss w.r.t. LGNN parameters using finite differences.
   Returns a vector of ΔW matrices parallel to params.theta/phi."""
function lgnn_grad_fd(params::LGNNParams, A, D, Ajs, BL, D_L, U, S, L_G, n, m;
                      eps::Float64=1e-4)
    function eval_loss(p)
        pr = lgnn_forward(p, A, D, Ajs, BL, D_L, U, S, n, m)
        return gnn_loss(pr, L_G)
    end
    # Numerical gradient for theta matrices
    dtheta = [zeros(size(w)) for w in params.theta]
    dphi   = [zeros(size(w)) for w in params.phi]
    dW_out = zeros(size(params.W_out))
    db_out = zeros(size(params.b_out))

    base_loss = eval_loss(params)

    # Only perturb a random subset to keep cost manageable (SPSA-style)
    for k in eachindex(params.theta)
        for i in CartesianIndices(params.theta[k])
            new_params = LGNNParams(
                [j==k ? (w = copy(params.theta[j]); w[i] += eps; w) : params.theta[j]
                 for j in eachindex(params.theta)],
                params.phi, params.J, params.feat_dim, params.T_iter,
                params.W_out, params.b_out)
            dtheta[k][i] = (eval_loss(new_params) - base_loss) / eps
        end
    end
    return dtheta, dphi, dW_out, db_out
end

# ── RYDBERG HARDWARE WARM-START ──────────────────────────────────────────────
# Beyond standard circuit warm-start (Egger et al. 2021), we propose a
# HARDWARE-NATIVE warm-start that encodes the GNN solution directly into the
# Rydberg detuning schedule. This is unique to neutral-atom platforms.

"""Compute GNN-informed detuning schedule for Rydberg warm-start.
   p_gnn: GNN output probabilities, edges: graph edge list, d: spacing.
   Returns Delta_WS[i] for each atom — this replaces the uniform 2*V_i
   detuning used in standard QAOA."""
function rydberg_warmstart_delta(p_gnn::Vector{Float64},
                                  atoms::AtomList,
                                  edges::Vector{Tuple{Int,Int}},
                                  n_atoms::Int)
    C6 = 2π*862690
    # Standard detuning: Delta_i = 2 * sum_j V_ij
    # Warm-start: rescale by (2*p_i - 1) to bias towards GNN solution
    # Atoms with p_i ≈ 1 (strongly in partition 0) get larger +Delta (Rydberg-blockade)
    # Atoms with p_i ≈ 0 (strongly in partition 1) get smaller/negative Delta
    pos = [atoms[i] for i in 1:n_atoms]
    Delta_base = zeros(n_atoms)
    for (i,j) in edges
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        V_ij = C6/(4*r2^3)
        Delta_base[i] += V_ij; Delta_base[j] += V_ij
    end
    # Warm-start factor: (2*p_i - 1) ∈ [-1, 1]
    # Clamp to avoid |Delta_WS| < 0.01 (would remove blockade)
    alpha = clamp.(2.0 .* p_gnn .- 1.0, -0.95, 0.95)
    Delta_WS = Delta_base .* (1.0 .+ alpha)   # range: [0.05, 1.95] × Delta_base
    return Delta_WS
end

function rydberg_warmstart_delta_graph(p_gnn::Vector{Float64},
                                        atoms::AtomList,
                                        edges::Vector{Tuple{Int,Int}},
                                        n_atoms::Int;
                                        Omega_ref::Float64=2π*4.0)
    # Graph-topology warm-start for non-geometric graphs (reg3, etc.)
    # Delta_i = -Omega_ref * mean_{j in N(i)}(2*p_j - 1)
    # Anti-alignment bias: pushes atom i toward opposite partition from its neighbours
    degree   = zeros(Int,     n_atoms)
    nbr_bias = zeros(Float64, n_atoms)
    for (i,j) in edges
        degree[i]   += 1;       degree[j]   += 1
        nbr_bias[i] += 2.0*p_gnn[j] - 1.0
        nbr_bias[j] += 2.0*p_gnn[i] - 1.0
    end
    Delta_WS = zeros(Float64, n_atoms)
    for i in 1:n_atoms
        d_i = degree[i] > 0 ? degree[i] : 1
        Delta_WS[i] = -Omega_ref * clamp(nbr_bias[i] / d_i, -0.95, 0.95)
    end
    return Delta_WS
end

# ══════════════════════════════════════════════════════════════════════════
# ── GDQC 风格的图核特征提取 (基于 Djellabi/Hecker/Acheche 2026) ─────────────
# 用 LGNN 学到的 p_i 替代论文里的原子质量,作为"节点属性"编码进局域失调场,
# 做一次很短的探测演化 (与完整 QAOA 调度完全分开),提取关联矩阵 C_ij(t),
# 再结合最短路径距离矩阵做分箱,得到一个跟图大小无关的固定长度特征向量 χ(G)。
# ══════════════════════════════════════════════════════════════════════════

"""最短路径距离矩阵 (Floyd–Warshall, O(n^3), 在你的图规模下完全够用)。"""
function shortest_path_matrix(n::Int, edges::Vector{Tuple{Int,Int}})
    INF = 10^6
    D = fill(INF, n, n)
    for i in 1:n; D[i,i] = 0; end
    for (i,j) in edges
        D[i,j] = 1; D[j,i] = 1
    end
    for k in 1:n, i in 1:n, j in 1:n
        if D[i,k] + D[k,j] < D[i,j]
            D[i,j] = D[i,k] + D[k,j]
        end
    end
    return D
end

"""GNN 偏置的探测演化: δ_i^G = δ0 * (2*p_gnn[i] - 1) ∈ [-δ0, δ0]。
   这是一段固定的、很短的演化 (默认 t_probe=0.4 µs, 与论文 Fig.1-2 的例子一致),
   跟完整的 QAOA γ/β 调度完全分开,只用来提取关联矩阵 C_ij(t)。
   返回 n_atoms × n_atoms 的关联矩阵 (对角线置 0, 因为分箱只用非对角元素,
   与论文 Fig.2 一致 —— 对角元素在阻塞机制下总是很小,不携带区分信息)。"""
function gdqc_probe_correlation(p_gnn::Vector{Float64}, atoms::AtomList,
                                 n_atoms::Int; delta0::Float64=2π*1.0,
                                 t_probe::Float64=0.4, Omega_probe::Float64=2π*1.0)
    delta_G = delta0 .* (2.0 .* p_gnn .- 1.0)
    Delta_wf = [piecewise_constant(clocks=[0.0, t_probe], values=[delta_G[i]])
                for i in 1:n_atoms]
    h = rydberg_h(atoms; Δ=Delta_wf,
                  Ω=piecewise_constant(clocks=[0.0, t_probe], values=[Omega_probe]))
    reg = zero_state(ComplexF64, n_atoms)
    prob = KrylovEvolution(reg, [0.0, t_probe], h)
    emulate!(prob)
    C = zeros(n_atoms, n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        C[i,j] = real(expect(put(n_atoms, i=>Op.n) * put(n_atoms, j=>Op.n), prob.reg))
    end
    return C
end

"""GDQC 特征向量 χ(C, D): 按关联值 (n_c_bins 个箱) × 图距离 (n_d_bins = 最大距离+1)
   做二维分箱, 展平成一维向量并做 L2 归一化。对应论文的算法1。"""
function gdqc_feature_vector(C::Matrix{Float64}, D::Matrix{Int};
                              n_c_bins::Int=10)
    n = size(C,1)
    finite_D = D[D .< 10^6]
    isempty(finite_D) && return zeros(n_c_bins)
    dmax = maximum(finite_D)
    n_d_bins = dmax + 1
    chi = zeros(n_c_bins * n_d_bins)
    dC = 1.0 / n_c_bins
    for i in 1:n, j in 1:n
        i == j && continue
        cval = clamp(C[i,j], 0.0, 1.0 - 1e-9)
        p_bin = min(n_c_bins - 1, floor(Int, cval / dC))
        d_bin = D[i,j]
        d_bin > dmax && continue
        idx = n_c_bins * d_bin + p_bin + 1
        chi[idx] += 1.0
    end
    nrm = norm(chi)
    return nrm > 0 ? chi ./ nrm : chi
end

"""完整流程: 图 + LGNN 概率 -> GDQC 特征向量 χ(G)。"""
function gdqc_signature(p_gnn::Vector{Float64}, atoms::AtomList,
                         edges::Vector{Tuple{Int,Int}}, n_atoms::Int;
                         n_c_bins::Int=10)
    C = gdqc_probe_correlation(p_gnn, atoms, n_atoms)
    D = shortest_path_matrix(n_atoms, edges)
    return gdqc_feature_vector(C, D; n_c_bins=n_c_bins)
end

"""两个特征向量之间的 GDQC 核值 (归一化内积, 取较短长度截断对齐,
   因为不同图的最大最短路径距离可能不同, 分箱后向量长度不同)。"""
function kappa_gdqc(chi1::Vector{Float64}, chi2::Vector{Float64})
    n = min(length(chi1), length(chi2))
    n == 0 && return 0.0
    return dot(chi1[1:n], chi2[1:n])
end

# ── 复用的 LGNN 快速训练封装 ──────────────────────────────────────────────
# gnn_train / gnn_qaoa 两个模式里各自写了一遍几乎相同的训练循环,这里抽成
# 一个函数, 方便在 gdqc_validate 模式里复用, 不影响原有两个模式的代码。
# [FIX] 原来 best_p/best_ar 追踪连续松弛值, 会严重低估真实离散cut, 现在改成
# 追踪并返回真实离散cut最优的p (跟 train_lgnn_quick_weighted 一致)。epoch数
# 从200提到300。
function train_lgnn_quick(edges::Vector{Tuple{Int,Int}}, n_atoms::Int;
                           n_epochs::Int=300, init_seed::Int=1)
    A_lg, D_lg, Ajs_lg = build_power_adj(n_atoms, edges, 3)
    BL_lg, DL_lg, U_lg, S_lg = build_line_graph(n_atoms, edges)
    L_G = graph_laplacian(n_atoms, edges)
    # [FIX] 跟 train_lgnn_quick_weighted 同样的修复: GNN参数初始化独立设种子,
    # 不再意外继承图构造消耗剩下的随机数流状态。
    Random.seed!(init_seed + 20000)
    params = init_lgnn(3, 32, 8)
    best_ar = 0.0
    best_ar_discrete = 0.0; best_p_discrete = fill(0.5, n_atoms)
    discrete_cut(pvec) = begin
        cut = 0
        for (u,v) in edges
            (pvec[u] > 0.5) != (pvec[v] > 0.5) && (cut += 1)
        end
        cut / length(edges)
    end
    eval_fn(p) = gnn_loss(
        lgnn_forward(p, A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg, n_atoms, length(edges)), L_G)
    for epoch in 1:n_epochs
        p_out = lgnn_forward(params, A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg,
                              n_atoms, length(edges))
        ar_e = -gnn_loss(p_out, L_G) / length(edges)
        ar_e > best_ar && (best_ar = ar_e)
        dar = discrete_cut(p_out)   # [新增] 真实离散cut
        dar > best_ar_discrete && (best_ar_discrete = dar; best_p_discrete = copy(p_out))
        dk = 0.05/(epoch^0.101)
        ak = 0.005/(epoch^0.602)
        params = spsa_update_all(params, dk, ak, eval_fn)   # [FIX] 同时扰动全部参数块
    end
    return params, best_p_discrete, best_ar_discrete
end

# ── 轻量级角度优化 (不需要完整 GSM+CSM GP-BO, 用多起点 Nelder-Mead 即可) ────
# 用于 gdqc_validate 里快速拿到每张图"自己的最优角度", 作为迁移实验的源角度。
# dim 很小 (2*p_val, 通常 6~10), 多起点 NM 足够, 不需要动用 build_spectral_gp
# 等完整 BO 机器 (那些留给正式的 pscale / pscale_chain 跑, 不在这里重复)。
function quick_best_angles(atoms, edges, n_atoms, p_val::Int;
                            n_starts::Int=8, nm_iters::Int=300)
    dim = 2 * p_val
    lo = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    halton_pts = halton_samples(n_starts, lo, hi; skip=17)
    best_y = 1000.0; best_x = halton_pts[1]
    for x0 in halton_pts
        r = optimize(x -> begin
                xs = clamp.(x, lo, hi)
                try
                    lv = loss_fn(xs, atoms, edges, n_atoms)
                    isfinite(lv) ? clamp(lv, -(length(edges)+1.0), 1000.0) : 1000.0
                catch; 1000.0; end
            end, x0, NelderMead(),
            Optim.Options(iterations=nm_iters, f_reltol=1e-7, show_trace=false))
        y = Optim.minimum(r); x = clamp.(Optim.minimizer(r), lo, hi)
        y < best_y && (best_y = y; best_x = x)
    end
    ar = -best_y / length(edges)
    return best_x, ar
end

"""GDQC 核库条目 —— 定义在 @everywhere 块里 (而不是某个具体 mode 的 if 块内!),
   这样 gdqc_validate / gnn_qaoa 等任何模式都能引用同一个类型, 互相读写
   同一批 .jls 库文件不会出现"类型未定义"的报错。"""
struct GDQCLibEntry
    graph_type::String
    n_atoms::Int
    d::Float64
    label::String
    n_edges::Int
    chi::Vector{Float64}
    best_bx::Vector{Float64}
    best_ar::Float64
    cold_ar::Float64
    gnn_ar::Float64      # [新增] GNN真正参与求解(loss_fn_warmstart)得到的ar, 跟best_ar并列对比
    atoms::AtomList
    edges::Vector{Tuple{Int,Int}}
end

"""GDQCLibEntry 的带权版本, 多一个 weights 字段。χ(G) 本身(gdqc_signature)
   继续只用拓扑距离 (不受 weights 影响, 见聊天里的讨论: GDQC 的距离轴设计上
   编码的是节点属性/拓扑关系, 不是边权重), 但 best_bx/best_ar/cold_ar 这些
   跟"求解质量"相关的量, 全部要用带权的评估函数才有意义。"""
struct GDQCLibEntryWeighted
    graph_type::String
    n_atoms::Int
    label::String
    n_edges::Int
    chi::Vector{Float64}
    best_bx::Vector{Float64}
    best_ar::Float64
    cold_ar::Float64
    gnn_ar::Float64      # [新增] 同上, 加权版
    atoms::AtomList
    edges::Vector{Tuple{Int,Int}}
    weights::Vector{Float64}
end

"""核信息引导的热启动: 在库里找 p_val 维度匹配的条目, 按 κ_GDQC 相似度
   取 top-k 加权平均它们的 best_bx, 作为新图的 BO 种子角度。
   similarity 低于阈值时, 调用方应该退回冷启动 —— 这里返回的
   ws.similarity 就是给调用方做阈值判断用的。"""
function kernel_warmstart(chi_new::Vector{Float64}, lib::Vector{GDQCLibEntry},
                           p_val::Int; k::Int=3)
    cands = filter(e -> length(e.best_bx) == 2*p_val, lib)
    isempty(cands) && return nothing
    sims = [kappa_gdqc(chi_new, e.chi) for e in cands]
    order = sortperm(sims, rev=true)
    kk = min(k, length(cands))
    topk = cands[order[1:kk]]
    topw = sims[order[1:kk]]
    sw = sum(topw)
    w = sw > 0 ? topw ./ sw : fill(1.0/kk, kk)
    angles0 = sum(w[i] .* topk[i].best_bx for i in eachindex(topk))
    return (angles0=angles0, similarity=maximum(sims))
end

"""随机角度基线: 抽 n_samples 组均匀随机角度, 直接用 loss_fn 评估 (不优化),
   取平均 AR。用来当"零样本迁移"的对照组 —— 如果迁移角度比这个基线好很多,
   说明迁移确实带来了信息, 不是白给的。"""
function random_angle_baseline(atoms, edges, n_atoms, p_val::Int; n_samples::Int=20)
    dim = 2 * p_val
    lo = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    ars = Float64[]
    for _ in 1:n_samples
        x = lo .+ rand(dim) .* (hi .- lo)
        try
            lv = loss_fn(x, atoms, edges, n_atoms)
            isfinite(lv) && push!(ars, -clamp(lv, -(length(edges)+1.0), 1000.0)/length(edges))
        catch; end
    end
    isempty(ars) && return 0.0
    return mean(ars)
end

"""quick_best_angles 的带权版本: 除了改用 loss_fn_weighted, 归一化用
   sum(weights) 而不是 length(edges)。"""
function quick_best_angles_weighted(atoms, edges, weights::Vector{Float64}, n_atoms,
                                     p_val::Int; n_starts::Int=8, nm_iters::Int=300)
    dim = 2 * p_val
    lo = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    halton_pts = halton_samples(n_starts, lo, hi; skip=17)
    best_y = 1000.0; best_x = halton_pts[1]
    total_w = sum(weights)
    for x0 in halton_pts
        r = optimize(x -> begin
                xs = clamp.(x, lo, hi)
                try
                    lv = loss_fn_weighted(xs, atoms, edges, weights, n_atoms)
                    isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
                catch; 1000.0; end
            end, x0, NelderMead(),
            Optim.Options(iterations=nm_iters, f_reltol=1e-7, show_trace=false))
        y = Optim.minimum(r); x = clamp.(Optim.minimizer(r), lo, hi)
        y < best_y && (best_y = y; best_x = x)
    end
    ar = -best_y / total_w
    return best_x, ar
end

"""random_angle_baseline 的带权版本。"""
function random_angle_baseline_weighted(atoms, edges, weights::Vector{Float64}, n_atoms,
                                         p_val::Int; n_samples::Int=20)
    dim = 2 * p_val
    lo = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    ars = Float64[]
    total_w = sum(weights)
    for _ in 1:n_samples
        x = lo .+ rand(dim) .* (hi .- lo)
        try
            lv = loss_fn_weighted(x, atoms, edges, weights, n_atoms)
            isfinite(lv) && push!(ars, -clamp(lv, -(total_w+1.0), 1000.0)/total_w)
        catch; end
    end
    isempty(ars) && return 0.0
    return mean(ars)
end

@everywhere begin
"""quick_best_angles 的 BO 版本 —— 跟 gnn_qaoa/compute_task_p 用的**同一套**
   GSM+CSM 光谱混合核贝叶斯优化(build_spectral_gp + map_spectral_hp +
   TS/EI/LCB 三阶段采集函数), 不是简化的多起点 NelderMead。NelderMead 仍然
   出现在循环里, 但只是 BO 主循环内每 refine_every 轮做一次局部精修
   (跟 gnn_qaoa 里的用法完全一致), 不是主驱动。"""
function quick_best_angles_bo(atoms, edges, n_atoms, p_val::Int;
                                           budget::Int=0, n_init::Int=0, seed::Int=42)
    dim = 2 * p_val
    lo  = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    budget = budget > 0 ? budget : max(200, 100*dim)
    n_init = n_init > 0 ? n_init : max(20, 8*dim)

    Random.seed!(seed)
    cache = Dict{Vector{Float64},Float64}()
    function ev!(x)
        xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
        haskey(cache, key) && return cache[key]
        v = try
            lv = loss_fn(xs, atoms, edges, n_atoms)
            isfinite(lv) ? clamp(lv, -(length(edges)+1.0), 1000.0) : 1000.0
        catch; 1000.0; end
        cache[key] = v; return v
    end

    halton_pts = halton_samples(n_init, lo, hi; skip=(seed*7)%100)
    init_xs = [clamp.(pt.+randn(dim).*0.05.*(hi.-lo), lo, hi) for pt in halton_pts]
    init_ys = Float64[ev!(x) for x in init_xs]
    X=hcat(init_xs...); y=copy(init_ys)
    by=minimum(init_ys); bx=init_xs[argmin(init_ys)]

    Q_g=2; Q_c=2
    n_cand        = dim>=8 ? 800 : 400
    map_every     = dim>=8 ? 12 : 18
    phase1_end    = dim>=8 ? 0.45 : 0.30
    phase2_end    = dim>=8 ? 0.78 : 0.68
    nr_hp         = dim>=8 ? 4 : 2
    stag_limit    = dim>=8 ? 25 : 20
    refine_every  = 15
    nm_iters      = dim>=8 ? 200 : 100
    refine_scales = dim>=8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
    θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp); stag=0
    n_bo_iter = max(budget - n_init, 0)

    for iter in 1:n_bo_iter
        t=iter/n_bo_iter
        iter>1 && iter%map_every==0 && (θ_gp=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp))
        gp=build_spectral_gp(X,y,θ_gp,Q_g,Q_c,dim)
        C=lo.+rand(dim,n_cand).*(hi.-lo)
        if t<phase1_end
            xn=C[:,ts_acq(gp,C;ns=10)]
        elseif t<phase2_end
            ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
        else
            β=max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
            μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]
        end
        yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
        if yn<by; by=yn; bx=copy(xn); stag=0
        else; t>=phase1_end && (stag+=1); end
        if stag>=stag_limit
            for _ in 1:2
                xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                X=hcat(X,xr); push!(y,yr); yr<by && (by=yr;bx=copy(xr))
            end; stag=0
        end
        if iter%refine_every==0 && t>=phase1_end
            for sc in refine_scales
                xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                           Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                yr<by-1e-6 && (by=yr;bx=xr;stag=0;X=hcat(X,xr);push!(y,yr))
            end
        end
    end
    ar = -by / length(edges)
    return bx, ar
end
end # @everywhere begin (quick_best_angles_bo)

@everywhere begin
"""quick_best_angles_bo 的带权版本, 逐行一致, 只换 loss_fn_weighted 和
   sum(weights) 归一化。"""
function quick_best_angles_bo_weighted(atoms, edges, weights::Vector{Float64},
                                                     n_atoms, p_val::Int;
                                                     budget::Int=0, n_init::Int=0, seed::Int=42)
    dim = 2 * p_val
    lo  = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    budget = budget > 0 ? budget : max(200, 100*dim)
    n_init = n_init > 0 ? n_init : max(20, 8*dim)
    total_w = sum(weights)

    Random.seed!(seed)
    cache = Dict{Vector{Float64},Float64}()
    function ev!(x)
        xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
        haskey(cache, key) && return cache[key]
        v = try
            lv = loss_fn_weighted(xs, atoms, edges, weights, n_atoms)
            isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
        catch; 1000.0; end
        cache[key] = v; return v
    end

    halton_pts = halton_samples(n_init, lo, hi; skip=(seed*7)%100)
    init_xs = [clamp.(pt.+randn(dim).*0.05.*(hi.-lo), lo, hi) for pt in halton_pts]
    init_ys = Float64[ev!(x) for x in init_xs]
    X=hcat(init_xs...); y=copy(init_ys)
    by=minimum(init_ys); bx=init_xs[argmin(init_ys)]

    Q_g=2; Q_c=2
    n_cand        = dim>=8 ? 800 : 400
    map_every     = dim>=8 ? 12 : 18
    phase1_end    = dim>=8 ? 0.45 : 0.30
    phase2_end    = dim>=8 ? 0.78 : 0.68
    nr_hp         = dim>=8 ? 4 : 2
    stag_limit    = dim>=8 ? 25 : 20
    refine_every  = 15
    nm_iters      = dim>=8 ? 200 : 100
    refine_scales = dim>=8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
    θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp); stag=0
    n_bo_iter = max(budget - n_init, 0)

    for iter in 1:n_bo_iter
        t=iter/n_bo_iter
        iter>1 && iter%map_every==0 && (θ_gp=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp))
        gp=build_spectral_gp(X,y,θ_gp,Q_g,Q_c,dim)
        C=lo.+rand(dim,n_cand).*(hi.-lo)
        if t<phase1_end
            xn=C[:,ts_acq(gp,C;ns=10)]
        elseif t<phase2_end
            ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
        else
            β=max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
            μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]
        end
        yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
        if yn<by; by=yn; bx=copy(xn); stag=0
        else; t>=phase1_end && (stag+=1); end
        if stag>=stag_limit
            for _ in 1:2
                xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                X=hcat(X,xr); push!(y,yr); yr<by && (by=yr;bx=copy(xr))
            end; stag=0
        end
        if iter%refine_every==0 && t>=phase1_end
            for sc in refine_scales
                xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                           Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                yr<by-1e-6 && (by=yr;bx=xr;stag=0;X=hcat(X,xr);push!(y,yr))
            end
        end
    end
    ar = -by / total_w
    return bx, ar
end
end # @everywhere begin (quick_best_angles_bo_weighted)

@everywhere begin
"""quick_best_angles_bo 的"GNN真正参与求解"版本 —— 跟 quick_best_angles_bo
   唯一的区别是 ev!() 调 loss_fn_warmstart(GNN热启动失调场偏置) 而不是
   loss_fn(冷启动)。用来回答"让GNN也参与角度优化, 到底有没有帮助"这个问题,
   跟 quick_best_angles_bo 的结果并列比较, 不互相替代 (原因见聊天记录:
   如果拿这个结果去替换 quick_best_angles_bo 作为"自身最优角度"的基准,
   会让 GDQC 核相似度的验证实验变成循环论证 —— κ_GDQC 和"最优角度"都来自
   同一个 p_gnn, 相关性可能只是反映"GNN输出像→热启动像", 而不是反映
   GDQC 探测演化 (关联矩阵) 本身带来了额外信息)。"""
function quick_best_angles_bo_gnn(atoms, edges, n_atoms, p_gnn::Vector{Float64},
                                               p_val::Int, graph_type::String;
                                               budget::Int=0, n_init::Int=0, seed::Int=42)
    dim = 2 * p_val
    lo  = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    budget = budget > 0 ? budget : max(200, 100*dim)
    n_init = n_init > 0 ? n_init : max(20, 8*dim)

    Random.seed!(seed)
    cache = Dict{Vector{Float64},Float64}()
    function ev!(x)
        xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
        haskey(cache, key) && return cache[key]
        v = try
            lv = loss_fn_warmstart(xs, p_gnn, atoms, edges, n_atoms; graph_type=graph_type)
            isfinite(lv) ? clamp(lv, -(length(edges)+1.0), 1000.0) : 1000.0
        catch; 1000.0; end
        cache[key] = v; return v
    end

    halton_pts = halton_samples(n_init, lo, hi; skip=(seed*7)%100)
    init_xs = [clamp.(pt.+randn(dim).*0.05.*(hi.-lo), lo, hi) for pt in halton_pts]
    init_ys = Float64[ev!(x) for x in init_xs]
    X=hcat(init_xs...); y=copy(init_ys)
    by=minimum(init_ys); bx=init_xs[argmin(init_ys)]

    Q_g=2; Q_c=2
    n_cand        = dim>=8 ? 800 : 400
    map_every     = dim>=8 ? 12 : 18
    phase1_end    = dim>=8 ? 0.45 : 0.30
    phase2_end    = dim>=8 ? 0.78 : 0.68
    nr_hp         = dim>=8 ? 4 : 2
    stag_limit    = dim>=8 ? 25 : 20
    refine_every  = 15
    nm_iters      = dim>=8 ? 200 : 100
    refine_scales = dim>=8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
    θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp); stag=0
    n_bo_iter = max(budget - n_init, 0)

    for iter in 1:n_bo_iter
        t=iter/n_bo_iter
        iter>1 && iter%map_every==0 && (θ_gp=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp))
        gp=build_spectral_gp(X,y,θ_gp,Q_g,Q_c,dim)
        C=lo.+rand(dim,n_cand).*(hi.-lo)
        if t<phase1_end
            xn=C[:,ts_acq(gp,C;ns=10)]
        elseif t<phase2_end
            ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
        else
            β=max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
            μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]
        end
        yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
        if yn<by; by=yn; bx=copy(xn); stag=0
        else; t>=phase1_end && (stag+=1); end
        if stag>=stag_limit
            for _ in 1:2
                xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                X=hcat(X,xr); push!(y,yr); yr<by && (by=yr;bx=copy(xr))
            end; stag=0
        end
        if iter%refine_every==0 && t>=phase1_end
            for sc in refine_scales
                xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                           Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                yr<by-1e-6 && (by=yr;bx=xr;stag=0;X=hcat(X,xr);push!(y,yr))
            end
        end
    end
    ar = -by / length(edges)
    return bx, ar
end
end # @everywhere begin (quick_best_angles_bo_gnn)

@everywhere begin
"""quick_best_angles_bo_gnn 的带权版本, 用 loss_fn_warmstart_weighted。"""
function quick_best_angles_bo_gnn_weighted(atoms, edges, weights::Vector{Float64},
                                                         n_atoms, p_gnn::Vector{Float64}, p_val::Int;
                                                         budget::Int=0, n_init::Int=0, seed::Int=42)
    dim = 2 * p_val
    lo  = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)
    budget = budget > 0 ? budget : max(200, 100*dim)
    n_init = n_init > 0 ? n_init : max(20, 8*dim)
    total_w = sum(weights)

    Random.seed!(seed)
    cache = Dict{Vector{Float64},Float64}()
    function ev!(x)
        xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
        haskey(cache, key) && return cache[key]
        v = try
            lv = loss_fn_warmstart_weighted(xs, p_gnn, atoms, edges, weights, n_atoms)
            isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
        catch; 1000.0; end
        cache[key] = v; return v
    end

    halton_pts = halton_samples(n_init, lo, hi; skip=(seed*7)%100)
    init_xs = [clamp.(pt.+randn(dim).*0.05.*(hi.-lo), lo, hi) for pt in halton_pts]
    init_ys = Float64[ev!(x) for x in init_xs]
    X=hcat(init_xs...); y=copy(init_ys)
    by=minimum(init_ys); bx=init_xs[argmin(init_ys)]

    Q_g=2; Q_c=2
    n_cand        = dim>=8 ? 800 : 400
    map_every     = dim>=8 ? 12 : 18
    phase1_end    = dim>=8 ? 0.45 : 0.30
    phase2_end    = dim>=8 ? 0.78 : 0.68
    nr_hp         = dim>=8 ? 4 : 2
    stag_limit    = dim>=8 ? 25 : 20
    refine_every  = 15
    nm_iters      = dim>=8 ? 200 : 100
    refine_scales = dim>=8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
    θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp); stag=0
    n_bo_iter = max(budget - n_init, 0)

    for iter in 1:n_bo_iter
        t=iter/n_bo_iter
        iter>1 && iter%map_every==0 && (θ_gp=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp))
        gp=build_spectral_gp(X,y,θ_gp,Q_g,Q_c,dim)
        C=lo.+rand(dim,n_cand).*(hi.-lo)
        if t<phase1_end
            xn=C[:,ts_acq(gp,C;ns=10)]
        elseif t<phase2_end
            ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
            μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
        else
            β=max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
            μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]
        end
        yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
        if yn<by; by=yn; bx=copy(xn); stag=0
        else; t>=phase1_end && (stag+=1); end
        if stag>=stag_limit
            for _ in 1:2
                xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                X=hcat(X,xr); push!(y,yr); yr<by && (by=yr;bx=copy(xr))
            end; stag=0
        end
        if iter%refine_every==0 && t>=phase1_end
            for sc in refine_scales
                xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                           Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                yr<by-1e-6 && (by=yr;bx=xr;stag=0;X=hcat(X,xr);push!(y,yr))
            end
        end
    end
    ar = -by / total_w
    return bx, ar
end
end # @everywhere begin (quick_best_angles_bo_gnn_weighted)

# ══════════════════════════════════════════════════════════════════════════
# ── 带权图工具函数: 生成带权 3-regular 图、把边权重
# 嵌入成二维原子坐标、带权 cut 的期望值/最优值计算。
# ══════════════════════════════════════════════════════════════════════════

@everywhere begin
"""带权随机 3-regular 图: 复用已有的 make_reg3_graph 拿拓扑, 再随机赋权重。
   权重单位是我们自己定义的"逻辑权重" w_jk —— 它和物理相互作用强度、失调场
   之间的换算关系在 weighted_graph_target_distances 里统一处理, 这里不用关心
   具体单位。
   [FIX] 原来定义在两个@everywhere代码块的空隙里, 只在主进程上有定义——
   之前所有用到它的模式(gnn_qaoa_weighted等)从来没真正触发过并行worker,
   pmap 一直在主进程上串行跑, 所以没暴露过; 这次给 gnn_pretrain_and_qaoa_
   weighted 真正加了并行worker(通过pmap在worker上调用
   make_reg3_weighted_graph_target3d, 它内部调这个函数), 才第一次报错
   UndefVarError。纳入@everywhere块修复。"""
function make_reg3_weighted_graph(n_atoms::Int, seed::Int=42;
                                   wmin::Float64=0.5, wmax::Float64=1.0)
    edges = make_reg3_graph(n_atoms, seed)
    # 显式 RNG(跨 Julia 版本稳定), 偏移量保证跟拓扑生成用不同的随机流
    rngw = MersenneTwister(seed + 1000)
    weights = wmin .+ rand(rngw, length(edges)) .* (wmax - wmin)
    return edges, weights
end

"""蜂窝(honeycomb)格点衍生图的核心构造 —— 拓扑**直接从蜂窝格点的邻接关系
   生成**, 不是先独立生成随机拓扑再摆到一个跟拓扑无关的格子上。这是论文
   Bauer et al. Method B 自己测"随机图"时用的方式("random graphs...
   generated from square and honeycomb lattices")——拓扑和位置从构造第一步
   就是自洽的, 不存在"边距离没对上"这回事(边全部严格等长=d)。
   [注意] 这个构造方式测的是"从格点生成的图", 不是"任意给定的随机3-regular
   图"——已经在聊天记录里确认过: 蜂窝格衍生的拓扑边全部等长, 套上加权场景
   会得到权重全部相等的退化结果, 本质上就是无权图。所以这里独立成一个通用
   的**无权**图类型, 用于检验"论文Method B自己的图生成方式", 不挂在
   gnn_qaoa_weighted 那条"加权随机3-regular图"的主线下面。
   蜂窝格用标准双子格构造: A子格 R_A(m,n)=m·a1+n·a2, B子格=R_A+δ1, 用
   δ1=(1,0)d, δ2=(-1/2,√3/2)d, δ3=(-1/2,-√3/2)d 三个近邻键矢量(120°夹角,
   长度都是d), a1=δ1-δ3, a2=δ1-δ2。A(m,n) 连接 B(m,n)、B(m,n-1)、B(m-1,n)。
   在有限patch里生成, 边界原子的邻居数会小于3(格点patch躲不开的边界效应,
   不是bug), 会如实打印实际度数分布, 不假装成精确3-regular。"""
function make_honeycomb_graph(n_atoms::Int, d::Float64)
    δ1 = (d, 0.0); δ2 = (-d/2, d*sqrt(3)/2); δ3 = (-d/2, -d*sqrt(3)/2)
    a1 = (δ1[1]-δ3[1], δ1[2]-δ3[2]); a2 = (δ1[1]-δ2[1], δ1[2]-δ2[2])

    # patch 尺寸留够余量 (2*M*N 个格点, 取够 n_atoms 个还有富余去挑高度数的)
    M = max(3, ceil(Int, sqrt(n_atoms)) + 2)
    N = M

    posA = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
    posB = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
    for m in 0:(M-1), n in 0:(N-1)
        RA = (m*a1[1]+n*a2[1], m*a1[2]+n*a2[2])
        posA[(m,n)] = RA
        posB[(m,n)] = (RA[1]+δ1[1], RA[2]+δ1[2])
    end

    # 汇总所有格点(A,B各(m,n)一个标签), 建邻接关系
    all_sites = Tuple{Symbol,Int,Int}[]
    for (m,n) in keys(posA); push!(all_sites, (:A,m,n)); end
    for (m,n) in keys(posB); push!(all_sites, (:B,m,n)); end
    site_pos = Dict{Tuple{Symbol,Int,Int},Tuple{Float64,Float64}}()
    for (m,n) in keys(posA); site_pos[(:A,m,n)] = posA[(m,n)]; end
    for (m,n) in keys(posB); site_pos[(:B,m,n)] = posB[(m,n)]; end

    adj = Dict{Tuple{Symbol,Int,Int},Vector{Tuple{Symbol,Int,Int}}}()
    for s in all_sites; adj[s] = Tuple{Symbol,Int,Int}[]; end
    for m in 0:(M-1), n in 0:(N-1)
        a = (:A,m,n)
        for b in [(:B,m,n), (:B,m,n-1), (:B,m-1,n)]
            haskey(site_pos, b) || continue
            push!(adj[a], b); push!(adj[b], a)
        end
    end

    # [FIX] 原来是"整个patch里独立按度数+离中心距离排序, 挑前n_atoms个"——
    # 这样选出来的原子即使各自在完整patch里度数是3, 也不代表它们互相之间
    # 是邻居, 选完之后大量边两端有一端没被选中, 直接被砍掉, 导致度数暴跌
    # (实测在n=10时度数均值只有2.00, 60%的原子度数<3)。改成从中心找一个
    # 种子原子, 沿着蜂窝格邻接关系做BFS(像滚雪球一样一格格往外长), 长满
    # n_atoms个为止——这样保证选出来的这一片天然连通、紧凑, 内部原子的3个
    # 邻居基本都在这一片里面, 只有最外层边界那圈才会度数不足。
    cx = mean([p[1] for p in values(site_pos)])
    cy = mean([p[2] for p in values(site_pos)])
    seed_site = argmin(s -> (site_pos[s][1]-cx)^2+(site_pos[s][2]-cy)^2, all_sites)

    chosen = Tuple{Symbol,Int,Int}[seed_site]
    chosen_set = Set([seed_site])
    frontier = copy(adj[seed_site])
    while length(chosen) < n_atoms && !isempty(frontier)
        # 每轮从当前"边界"(已选原子的、还没被选中的邻居)里挑离中心最近的
        sort!(frontier, by = s -> (site_pos[s][1]-cx)^2+(site_pos[s][2]-cy)^2)
        next_s = frontier[1]
        deleteat!(frontier, 1)
        next_s in chosen_set && continue
        push!(chosen, next_s); push!(chosen_set, next_s)
        for t in adj[next_s]
            t in chosen_set || push!(frontier, t)
        end
    end
    if length(chosen) < n_atoms
        # patch不够大(理论上不会发生, M/N留了余量), 兜底报错而不是静默返回不够的原子数
        error("蜂窝格patch不够大, 只长出了$(length(chosen))个原子, 需要$(n_atoms)个")
    end
    idx_of = Dict(s => i for (i,s) in enumerate(chosen))

    edges = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for s in chosen
        for t in adj[s]
            t in chosen_set || continue
            i, j = idx_of[s], idx_of[t]
            key = (min(i,j), max(i,j))
            key in seen && continue
            push!(seen, key); push!(edges, key)
        end
    end

    positions = [(site_pos[s][1], site_pos[s][2], 0.0) for s in chosen]
    atoms = AtomList(positions)

    degs = zeros(Int, length(chosen))
    for (i,j) in edges; degs[i]+=1; degs[j]+=1; end
    @printf("  [诊断] 蜂窝格衍生图度数分布: min=%d  max=%d  mean=%.2f  (度数=3的原子占比 %.1f%%)\n",
            minimum(degs), maximum(degs), mean(degs), 100*count(==(3),degs)/length(degs))

    return atoms, edges
end

"""make_honeycomb_graph 的加权包装 —— 权重从蜂窝格键长(恒为d)直接读出来,
   全部相等 (见 make_honeycomb_graph 文档字符串: 这本质上是退化成无权的
   加权图, 保留这个包装只是为了兼容以前的调用方式)。"""
function make_honeycomb_graph_weighted(n_atoms::Int, d::Float64)
    atoms, edges = make_honeycomb_graph(n_atoms, d)
    positions = [atoms[i] for i in 1:length(atoms)]
    weights = Float64[]
    for (i,j) in edges
        xi,yi,_ = positions[i]; xj,yj,_ = positions[j]
        r2 = (xi-xj)^2 + (yi-yj)^2
        push!(weights, C6_VAL/(4.0*r2^3))
    end
    return atoms, edges, weights
end

"""蜂窝格 + 受控随机抖动 —— 解决"独立随机拓扑硬塞进几何"这整类方案共同的
   根源问题(2n个自由度要同时满足O(n)条边"要近"+O(n²)对非边"要远"的约束,
   n越大越难, 不管换哪种损失函数都绕不开), 不是在同一个框架里继续调参数。
   核心思路: 拓扑固定用蜂窝格天然邻接关系(跟3-regular天生吻合, 边界效应
   跟 make_honeycomb_graph 一样, 会打印度数分布), 位置在蜂窝格点基础上加
   一点小幅度、独立同分布的高斯随机位移(标准差=jitter_frac*d)。拓扑和
   位置从来没有分离过, 不存在需要retrofit的问题, 也就没有embedding残差
   这个概念(没有优化步骤, 直接确定性摆位置, 只要 jitter_frac 足够小,
   拓扑不会因为抖动而改变)。权重的随机性和差异性直接来自抖动本身,
   jitter_frac 越大, 权重差异越大, 但也会增大 unwanted_ratio(抖动太猛
   可能把某些非边意外抖到比某些边还近), 需要在两者之间找平衡, 默认0.15
   (键长的15%)是一个温和的起点。"""
function make_honeycomb_graph_jittered(n_atoms::Int, d::Float64;
                                        jitter_frac::Float64=0.15, seed::Int=42)
    atoms0, edges = make_honeycomb_graph(n_atoms, d)
    pos0 = [atoms0[i] for i in 1:length(atoms0)]
    n_at = length(pos0)

    Random.seed!(seed + 7000)
    σ = jitter_frac * d
    positions = [(pos0[i][1]+randn()*σ, pos0[i][2]+randn()*σ, 0.0) for i in 1:n_at]
    atoms = AtomList(positions)

    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    weights = Float64[]
    edge_sum = 0.0; nonedge_sum = 0.0
    for (i,j) in edges
        xi,yi,_ = positions[i]; xj,yj,_ = positions[j]
        r2 = (xi-xj)^2 + (yi-yj)^2
        w = C6_VAL/(4.0*r2^3)
        push!(weights, w); edge_sum += w
    end
    for i in 1:n_at, j in (i+1):n_at
        (i,j) in edge_set && continue
        xi,yi,_ = positions[i]; xj,yj,_ = positions[j]
        r2 = (xi-xj)^2 + (yi-yj)^2
        nonedge_sum += C6_VAL/(4.0*r2^3)
    end
    unwanted_ratio = edge_sum > 0 ? nonedge_sum/edge_sum : NaN
    return atoms, edges, weights, unwanted_ratio
end

"""贪心最近邻的随机几何3-regular图 —— 综合各家方法的优点, 不是照搬任何一篇
   论文, 而是从几个方案里取长补短拼出来的:
     - 从 quantum wire(Kombe et al. 2025)反过来吸取的教训: 那篇论文能高效
       处理的是"大部分边天然近邻、只有少数长程边"的图; 随机3-regular图
       几乎每条边都是"长程"的, 硬给它搭线开销会爆炸——**与其事后给随机
       拓扑硬凑几何(target/force那条路), 不如反过来让拓扑从几何生成时就
       天然避免长程边**。
     - 保留 jitter(蜂窝格+抖动)"零retrofit, 位置决定权重"这个优点, 但
       去掉它"拓扑被蜂窝格锁死, 不是真正随机"这个缺点。
     - 不依赖固定格点摆位置, 不会有蜂窝格那种小n时边界效应主导度数分布
       的问题(贪心匹配漏掉的原子是偶发的, 不是系统性的)。
   具体做法: 先在2D随机撒点(scatter_points_2d, 带最小间距防止物理碰撞);
   把所有原子对按距离从近到远排序, 贪心地依次连边——只要两端度数都还没到
   3就连上——直到每个原子度数=3或者没有更多可连的对为止。极少数原子可能
   因为"最近的几个候选都已经被别人占满"卡在度数<3, 会如实打印实际度数
   分布, 不假装是精确3-regular。权重直接从贪心匹配后的真实距离读出。"""
function scatter_points_2d(n_atoms::Int, d::Float64; min_sep_frac::Float64=0.3,
                            seed::Int=42, max_tries::Int=200)
    Random.seed!(seed + 8000)
    canvas_R = 0.524 * d * sqrt(n_atoms)   # 沿用力场版同一套六边形密堆积推导
    min_sep = min_sep_frac * d
    positions = Tuple{Float64,Float64}[]
    for i in 1:n_atoms
        placed = false
        for _try in 1:max_tries
            x = (rand()-0.5) * 2.0 * canvas_R
            y = (rand()-0.5) * 2.0 * canvas_R
            ok = true
            for (px,py) in positions
                if (x-px)^2+(y-py)^2 < min_sep^2
                    ok = false; break
                end
            end
            if ok
                push!(positions, (x,y)); placed = true; break
            end
        end
        # 兜底: max_tries次都没找到合适位置(极少见), 放弃最小间距约束直接摆
        placed || push!(positions, (canvas_R*2*(rand()-0.5), canvas_R*2*(rand()-0.5)))
    end
    return positions
end

function make_reg3_graph_geometric_greedy(n_atoms::Int, d::Float64, seed::Int=42;
                                           min_sep_frac::Float64=0.6,
                                           max_match_frac::Float64=1.8)
    # [FIX] 原来只在撒点时防碰撞(min_sep_frac=0.3), 贪心匹配阶段对距离没有
    # 上限——最早匹配的几对可能凑巧撒得特别近, 排到后面的原子"剩菜"只能
    # 匹配远得多的对象, 六次方一放大权重比值直接失控(实测出现过40万倍)。
    # 现在给匹配阶段同时加上限和下限: min_sep_frac从0.3提到0.6(减少撒点
    # 阶段就出现极近对的概率), 新增 max_match_frac=1.8 (匹配阶段直接跳过
    # 超过 1.8×d 的候选对, 哪怕两端都还需要度数)。代价是可能有原子因为
    # "附近的候选都用完了、更远的又超出上限"而卡在度数<3, 会如实打印实际
    # 度数分布, 这是有意为之的权衡(宁可度数分布打点折扣, 也不接受权重比值
    # 失控)。
    positions = scatter_points_2d(n_atoms, d; min_sep_frac=min_sep_frac, seed=seed)

    pairs = Tuple{Int,Int,Float64}[]
    max_match_d2 = (max_match_frac * d)^2
    for i in 1:n_atoms, j in (i+1):n_atoms
        xi,yi = positions[i]; xj,yj = positions[j]
        r2 = (xi-xj)^2+(yi-yj)^2
        r2 <= max_match_d2 && push!(pairs, (i, j, r2))
    end
    sort!(pairs, by = p -> p[3])

    degree = zeros(Int, n_atoms)
    edges = Tuple{Int,Int}[]
    for (i,j,_) in pairs
        if degree[i] < 3 && degree[j] < 3
            push!(edges, (i,j))
            degree[i] += 1; degree[j] += 1
        end
    end

    atoms = AtomList([(positions[i][1], positions[i][2], 0.0) for i in 1:n_atoms])

    @printf("  [诊断] 贪心几何3-regular图度数分布: min=%d  max=%d  mean=%.2f  (度数=3的原子占比 %.1f%%)\n",
            minimum(degree), maximum(degree), mean(degree), 100*count(==(3),degree)/n_atoms)

    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    weights = Float64[]
    edge_sum = 0.0; nonedge_sum = 0.0
    for (i,j) in edges
        xi,yi = positions[i]; xj,yj = positions[j]
        r2 = (xi-xj)^2 + (yi-yj)^2
        w = C6_VAL/(4.0*r2^3)
        push!(weights, w); edge_sum += w
    end
    for i in 1:n_atoms, j in (i+1):n_atoms
        (i,j) in edge_set && continue
        xi,yi = positions[i]; xj,yj = positions[j]
        r2 = (xi-xj)^2 + (yi-yj)^2
        nonedge_sum += C6_VAL/(4.0*r2^3)
    end
    unwanted_ratio = edge_sum > 0 ? nonedge_sum/edge_sum : NaN

    return atoms, edges, weights, unwanted_ratio
end

"""scatter_points_2d 的三维版本, 用球体密堆积(FCC/HCP, 堆积效率η≈0.7405)
   推导画布半径, 不是照抄二维那套六边形密堆积(η≈0.9069)的公式。"""
function scatter_points_3d(n_atoms::Int, d::Float64; min_sep_frac::Float64=0.3,
                            seed::Int=42, max_tries::Int=200)
    rng = MersenneTwister(seed + 9000)      # 显式 RNG, 跨 Julia 版本可复现
    η = 0.7405
    canvas_R = (d/2.0) * (n_atoms/η)^(1/3)
    min_sep = min_sep_frac * d
    positions = Tuple{Float64,Float64,Float64}[]
    for i in 1:n_atoms
        placed = false
        for _try in 1:max_tries
            x = (rand(rng)-0.5) * 2.0 * canvas_R
            y = (rand(rng)-0.5) * 2.0 * canvas_R
            z = (rand(rng)-0.5) * 2.0 * canvas_R
            ok = true
            for (px,py,pz) in positions
                if (x-px)^2+(y-py)^2+(z-pz)^2 < min_sep^2
                    ok = false; break
                end
            end
            if ok
                push!(positions, (x,y,z)); placed = true; break
            end
        end
        placed || push!(positions, (canvas_R*2*(rand(rng)-0.5), canvas_R*2*(rand(rng)-0.5), canvas_R*2*(rand(rng)-0.5)))
    end
    return positions
end

"""make_reg3_graph_geometric_greedy 的三维版本 —— 逻辑完全一样(贪心按距离
   从近到远连边直到度数=3或超过max_match_frac上限), 唯一区别是自由度从2n
   变成3n, 用 scatter_points_3d 撒点。三维给了每个原子多一维空间去同时
   满足"边要近、非边要远"这两类约束, 理论上应该比二维更容易凑出规整的
   拓扑, 同时权重比值不失控。"""
function make_reg3_graph_geometric_greedy_3d(n_atoms::Int, d::Float64, seed::Int=42;
                                              min_sep_frac::Float64=0.6,
                                              max_match_frac::Float64=1.8)
    positions = scatter_points_3d(n_atoms, d; min_sep_frac=min_sep_frac, seed=seed)

    pairs = Tuple{Int,Int,Float64}[]
    max_match_d2 = (max_match_frac * d)^2
    for i in 1:n_atoms, j in (i+1):n_atoms
        xi,yi,zi = positions[i]; xj,yj,zj = positions[j]
        r2 = (xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        r2 <= max_match_d2 && push!(pairs, (i, j, r2))
    end
    sort!(pairs, by = p -> p[3])

    degree = zeros(Int, n_atoms)
    edges = Tuple{Int,Int}[]
    for (i,j,_) in pairs
        if degree[i] < 3 && degree[j] < 3
            push!(edges, (i,j))
            degree[i] += 1; degree[j] += 1
        end
    end

    atoms = AtomList(positions)

    @printf("  [诊断] 贪心几何3-regular图(三维)度数分布: min=%d  max=%d  mean=%.2f  (度数=3的原子占比 %.1f%%)\n",
            minimum(degree), maximum(degree), mean(degree), 100*count(==(3),degree)/n_atoms)

    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    weights = Float64[]
    edge_sum = 0.0; nonedge_sum = 0.0
    for (i,j) in edges
        xi,yi,zi = positions[i]; xj,yj,zj = positions[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        w = C6_VAL/(4.0*r2^3)
        push!(weights, w); edge_sum += w
    end
    for i in 1:n_atoms, j in (i+1):n_atoms
        (i,j) in edge_set && continue
        xi,yi,zi = positions[i]; xj,yj,zj = positions[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        nonedge_sum += C6_VAL/(4.0*r2^3)
    end
    unwanted_ratio = edge_sum > 0 ? nonedge_sum/edge_sum : NaN

    return atoms, edges, weights, unwanted_ratio
end

"""embed_reg3_force_2d 的三维版本(只保留 legacy k_mode, 三轮实测里最稳的
   那个, global那版没有必要一起搬三维)。逻辑完全一样(两个独立常数:
   k_rep=C6/4真实物理值, k_attr由孤立边对平衡点反解), 唯一区别是自由度
   从2n变成3n, 撒点范围用球体密堆积公式(跟 scatter_points_3d 同一套)。"""
function embed_reg3_force_3d(edges, n_atoms::Int;
                              d_target::Float64=12.0,
                              seed::Int=42, iters::Int=-1)
    # [FIX] 跟 embed_reg3_force_2d 同样的问题: iters固定不随n_atoms变化,
    # 大图收敛质量会随之恶化。同样按n_atoms²动态放大。
    actual_iters = iters > 0 ? iters : max(2000, round(Int, 20 * n_atoms^2))
    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    k_rep_legacy  = C6_VAL / 4.0
    k_attr_legacy = 3.0 * k_rep_legacy / d_target^8

    Random.seed!(seed + 6000)
    η = 0.7405
    canvas_R = (d_target/2.0) * (n_atoms/η)^(1/3)
    x0 = (rand(3*n_atoms) .- 0.5) .* (2.0 * canvas_R)

    function potential(x::Vector{Float64})
        V = 0.0
        for i in 1:n_atoms, j in (i+1):n_atoms
            dx = x[3i-2]-x[3j-2]; dy = x[3i-1]-x[3j-1]; dz = x[3i]-x[3j]
            d2 = dx^2 + dy^2 + dz^2 + 1e-6
            is_edge = (i,j) in edge_set
            V += k_rep_legacy / d2^3
            is_edge && (V += k_attr_legacy * d2)
        end
        return V
    end

    init_val = potential(x0)
    x_best = x0; best_val = init_val
    try
        r = optimize(potential, x0, LBFGS(), Optim.Options(iterations=actual_iters))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "reg3三维力场嵌入: LBFGS失败, 退回NelderMead" exception=e
    end
    try
        r = optimize(potential, x_best, NelderMead(), Optim.Options(iterations=actual_iters*2))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "reg3三维力场嵌入: NelderMead精修失败" exception=e
    end

    edge_dists = Float64[]
    nonedge_dists = Float64[]
    for i in 1:n_atoms, j in (i+1):n_atoms
        dx = x_best[3i-2]-x_best[3j-2]; dy = x_best[3i-1]-x_best[3j-1]; dz = x_best[3i]-x_best[3j]
        d  = sqrt(dx^2 + dy^2 + dz^2)
        if (i,j) in edge_set
            push!(edge_dists, d)
        else
            push!(nonedge_dists, d)
        end
    end
    @printf("  [诊断] 三维力场嵌入(legacy) 势能: 初始=%.6g → 最终=%.6g\n", init_val, best_val)
    @printf("  [诊断] 边距离(目标平衡点≈%.2f): min=%.2f  max=%.2f  mean=%.2f\n",
            d_target, minimum(edge_dists), maximum(edge_dists), mean(edge_dists))
    @printf("  [诊断] 非边距离: min=%.2f  max=%.2f  mean=%.2f\n",
            minimum(nonedge_dists), maximum(nonedge_dists), mean(nonedge_dists))

    positions = [(x_best[3i-2], x_best[3i-1], x_best[3i]) for i in 1:n_atoms]
    return AtomList(positions), best_val
end

"""带权随机 3-regular 图, 用 embed_reg3_force_3d 的三维力场嵌入(legacy),
   权重从收敛后的实际距离读出。"""
function make_reg3_weighted_graph_force_3d(n_atoms::Int, seed::Int=42;
                                            d_target::Float64=12.0)
    edges = make_reg3_graph(n_atoms, seed)
    atoms, stress_val = embed_reg3_force_3d(edges, n_atoms; d_target=d_target, seed=seed)
    pos = [atoms[i] for i in 1:n_atoms]
    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    weights = Float64[]
    edge_coupling_sum = 0.0
    nonedge_coupling_sum = 0.0
    for (i,j) in edges
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        w = C6_VAL/(4.0*r2^3)
        push!(weights, w)
        edge_coupling_sum += w
    end
    for i in 1:n_atoms, j in (i+1):n_atoms
        (i,j) in edge_set && continue
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        nonedge_coupling_sum += C6_VAL/(4.0*r2^3)
    end
    unwanted_ratio = edge_coupling_sum > 0 ? nonedge_coupling_sum/edge_coupling_sum : NaN
    return atoms, edges, weights, stress_val, unwanted_ratio
end

"""连续力场版三维嵌入, 但每条边有**自己专属**的目标平衡距离(从独立指定的
   目标权重反解出来), 不是像 embed_reg3_force_3d 那样所有边共用同一个
   d_target。这是"target模式的意图"(权重能独立指定、有真实差异)加上
   "force3d的实现能力"(连续力场在三维空间下收敛质量明显更好, 不是2D
   MDS嵌入那种一遇到随机拓扑就大幅失真的写法)拼在一起——2D的target之所以
   失败, 是2D+旧MDS嵌入算法在随机拓扑上收敛不动(甚至权重方向都会反),
   不是"逐边指定目标"这个思路本身有问题; 三维力场已经证明能大幅缓解收敛
   困难, 值得把逐边目标重新接回来试。
   排斥项(所有原子对, 真实C6/4)不变; 引力项跟legacy k_mode同样的物理推导
   (孤立边对令dV/dd=0解出k_attr), 只是**每条边各自反解**——k_attr_i =
   3·k_rep/(r*_i)^8, 其中 r*_i=(C6/(4·w*_i))^(1/6) 是这条边由目标权重
   w*_i 反推出来的目标距离。"""
function embed_reg3_force_3d_target(edges, n_atoms::Int, target_weights::Vector{Float64};
                                     seed::Int=42, iters::Int=-1)
    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    edge_idx = Dict((min(i,j),max(i,j)) => idx for (idx,(i,j)) in enumerate(edges))

    k_rep_legacy = C6_VAL / 4.0
    r_target = [(C6_VAL/(4.0*target_weights[idx]))^(1/6) for idx in 1:length(edges)]
    k_attr_per_edge = [3.0*k_rep_legacy/r_target[idx]^8 for idx in 1:length(edges)]
    d_scale = mean(r_target)   # 用平均目标距离定初始撒点范围, 跟legacy版同一套密堆积逻辑

    actual_iters = iters > 0 ? iters : max(2000, round(Int, 20 * n_atoms^2))

    rng = MersenneTwister(seed + 6000)      # 显式 RNG, 跨 Julia 版本可复现
    η = 0.7405
    canvas_R = (d_scale/2.0) * (n_atoms/η)^(1/3)
    x0 = (rand(rng, 3*n_atoms) .- 0.5) .* (2.0 * canvas_R)

    function potential(x::Vector{Float64})
        V = 0.0
        for i in 1:n_atoms, j in (i+1):n_atoms
            dx = x[3i-2]-x[3j-2]; dy = x[3i-1]-x[3j-1]; dz = x[3i]-x[3j]
            d2 = dx^2 + dy^2 + dz^2 + 1e-6
            V += k_rep_legacy / d2^3
            key = (i,j)
            key in edge_set && (V += k_attr_per_edge[edge_idx[key]] * d2)
        end
        return V
    end

    init_val = potential(x0)
    x_best = x0; best_val = init_val
    try
        r = optimize(potential, x0, LBFGS(), Optim.Options(iterations=actual_iters))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "reg3三维逐边目标力场嵌入: LBFGS失败, 退回NelderMead" exception=e
    end
    try
        r = optimize(potential, x_best, NelderMead(), Optim.Options(iterations=actual_iters*2))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "reg3三维逐边目标力场嵌入: NelderMead精修失败" exception=e
    end

    edge_dists = Float64[]
    nonedge_dists = Float64[]
    for i in 1:n_atoms, j in (i+1):n_atoms
        dx = x_best[3i-2]-x_best[3j-2]; dy = x_best[3i-1]-x_best[3j-1]; dz = x_best[3i]-x_best[3j]
        d  = sqrt(dx^2 + dy^2 + dz^2)
        if (i,j) in edge_set
            push!(edge_dists, d)
        else
            push!(nonedge_dists, d)
        end
    end
    @printf("  [诊断] 三维逐边目标力场嵌入 势能: 初始=%.6g → 最终=%.6g\n", init_val, best_val)
    @printf("  [诊断] 边距离(各自目标区间 [%.2f,%.2f]): min=%.2f  max=%.2f  mean=%.2f\n",
            minimum(r_target), maximum(r_target), minimum(edge_dists), maximum(edge_dists), mean(edge_dists))
    @printf("  [诊断] 非边距离: min=%.2f  max=%.2f  mean=%.2f\n",
            minimum(nonedge_dists), maximum(nonedge_dists), mean(nonedge_dists))

    positions = [(x_best[3i-2], x_best[3i-1], x_best[3i]) for i in 1:n_atoms]
    return AtomList(positions), best_val
end

"""带权随机 3-regular 图, target3d 版本 —— 先用 make_reg3_weighted_graph
   独立生成一组有真实差异的目标权重(跟原来的target模式一样, wmin~wmax),
   再用 embed_reg3_force_3d_target 逐边反解目标距离、三维力场嵌入, 最后
   权重仍然从收敛后的*实际*距离读出来(跟其它 force/jitter/greedy 系列
   一样, 位置决定权重是最终来源, 目标权重只是用来引导嵌入优化的意图,
   不是直接拿来用的最终数值)。"""
function make_reg3_weighted_graph_target3d(n_atoms::Int, seed::Int=42;
                                            wmin::Float64=0.5, wmax::Float64=1.0)
    edges, target_weights = make_reg3_weighted_graph(n_atoms, seed; wmin=wmin, wmax=wmax)
    atoms, stress_val = embed_reg3_force_3d_target(edges, n_atoms, target_weights; seed=seed)
    pos = [atoms[i] for i in 1:n_atoms]
    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    weights = Float64[]
    edge_coupling_sum = 0.0
    nonedge_coupling_sum = 0.0
    for (i,j) in edges
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        w = C6_VAL/(4.0*r2^3)
        push!(weights, w)
        edge_coupling_sum += w
    end
    for i in 1:n_atoms, j in (i+1):n_atoms
        (i,j) in edge_set && continue
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        nonedge_coupling_sum += C6_VAL/(4.0*r2^3)
    end
    unwanted_ratio = edge_coupling_sum > 0 ? nonedge_coupling_sum/edge_coupling_sum : NaN
    fidelity = length(weights) >= 3 ? cor(target_weights, weights) : NaN
    return atoms, edges, weights, stress_val, unwanted_ratio, fidelity
end

"""带权随机 3-regular 图, 位置先定、权重从实际距离读出 (跟上面
   make_reg3_weighted_graph 方向相反)。上面那个版本先指定目标权重, 再让
   embed_weighted_graph_2d 去凑位置, 对随机拓扑几乎总有残差(stress) —— 这
   是"matchstick graph"这类几何约束问题天生的困难, 换n或换种子都解决不了。
   [注意] 这个函数保留作为历史/对照参照, 已经不是 gnn_qaoa_weighted 的
   "geometric"选项默认调用的函数了 —— 那个现在换成了更贴合论文Method B、
   拓扑从格点邻接关系直接生成的 make_honeycomb_graph_weighted。这个函数
   的问题在于"位置"和"拓扑"从一开始就是脱节的两回事(方格内部度数是4,
   跟随机生成的3-regular拓扑无关), 见 make_honeycomb_graph_weighted 的
   文档字符串。"""
function make_reg3_weighted_graph_geometric(n_atoms::Int, d::Float64, seed::Int=42)
    edges = make_reg3_graph(n_atoms, seed)
    atoms = make_grid_atoms(n_atoms, d)
    pos = [atoms[i] for i in 1:n_atoms]
    weights = Float64[]
    for (i,j) in edges
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        push!(weights, C6_VAL/(4.0*r2^3))
    end
    return atoms, edges, weights
end

"""带权随机 3-regular 图, 连续力场版嵌入, 支持两种 k 的构造方式做对比:

   k_mode="global": 论文 Eq.6-7 的做法 —— 排斥项和引力项**共用同一个 k**,
     V = Σ_{i<j} (k/d_ij)^6 (所有原子对) + Σ_{(i,j)∈E} (d_ij/k)^2 (只有边)。
     跟原来的版本不同: 原来是两个独立常数(k_rep固定=C6/4, k_attr由孤立边
     对反解), 这个版本严格按论文"排斥引力共用一个k"的结构, k 本身就是
     "最优距离"的直接体现(k越大, 平衡点越远)。

   k_mode="legacy": 上一版的做法, 两个独立常数(k_rep=C6/4真实物理值,
     k_attr由孤立边对平衡点反解出来), 不是论文Eq.6-8的结构, 保留作为
     对照。实测两轮(不同初始撒点范围)结果完全一致, 残差和权重比值也是
     两种里最健康的, 目前是推荐默认。

   [已删除] k_mode="local"(论文Eq.8, 按共享三角形数量算每条边的k_ij):
     实测两轮权重比值都炸到四位数(一条边权重比其它边高出几千倍)——
     根源不是参数没调好, 是这套机制的设计目标(局部结构该有差异化的
     目标距离)跟"位置直接决定权重"这个哲学天生冲突: 只要边与边之间
     有意义的距离差异, 交给 w=C6/r⁶ 直接读权重, 六次方必然把差异放大到
     失控, 加下限只能防最坏情况、防不住这个根本问题。如果以后想让
     Eq.8这套机制真正发挥作用, 得放到 target 模式那种"权重独立指定、
     位置只是去近似"的框架里, 不适合这里。

   不管哪种 k_mode, 排斥项对所有原子对都有(天然防碰撞), 引力项只有边有;
   权重仍然是从收敛后的实际距离直接读出来的。

   [FIX] 初始撒点范围: 论文Eq.6的 k∝√(Area/n) 是"画布大小固定、n越多越挤"
   这个场景专用的公式, 我们做的是自由空间优化, 没有固定画布, 字面照搬不
   成立。真正该随n_atoms变化的不是k本身(k是局部量: 不管总共多少原子,
   每个原子永远只有3个"该近"的邻居, 跟n无关), 而是整个系统该占多大空间。
   用六边形密堆积算: 每个原子占据半径~k/2的"个人空间", n个这样的圆盘
   在2D密堆积(堆积效率η≈0.9069)需要面积 Area≈n·π(k/2)²/η≈0.865·n·k²,
   对应画布半径 R=√(Area/π)≈0.524·k·√n —— 这个用来定初始撒点范围, 不是
   用来定k本身。"""
function embed_reg3_force_2d(edges, n_atoms::Int;
                              k_mode::String="legacy",
                              d_target::Float64=12.0,
                              seed::Int=42, iters::Int=-1)
    # [FIX] 原来iters固定=2000, 不管n_atoms多大都一样——非边约束数量是
    # O(n²)增长的(n=10时只有30对, n=22时有231对), 同样的迭代预算面对更多
    # 约束, 收敛质量必然下降(实测n=22时unwanted_ratio从0.17级别恶化到0.98,
    # 边和非边距离完全重叠)。iters<0(没有显式指定)时, 按n_atoms²动态放大,
    # 以n=10时的2000为基准校准比例, n=22时会自动放大到约9680。
    actual_iters = iters > 0 ? iters : max(2000, round(Int, 20 * n_atoms^2))
    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    k_global = d_target   # global模式/非边排斥项/legacy模式统一用这个

    k_rep_legacy  = C6_VAL / 4.0
    k_attr_legacy = 3.0 * k_rep_legacy / d_target^8

    Random.seed!(seed + 6000)
    # [FIX] 六边形密堆积推出的画布半径, 替代原来随手设的 1.5 倍
    canvas_R = 0.524 * d_target * sqrt(n_atoms)
    x0 = (rand(2*n_atoms) .- 0.5) .* (2.0 * canvas_R)

    function potential(x::Vector{Float64})
        V = 0.0
        for i in 1:n_atoms, j in (i+1):n_atoms
            dx = x[2i-1]-x[2j-1]; dy = x[2i]-x[2j]
            d2 = dx^2 + dy^2 + 1e-6
            key = (i,j)
            is_edge = key in edge_set
            if k_mode == "legacy"
                V += k_rep_legacy / d2^3
                is_edge && (V += k_attr_legacy * d2)
            else
                V += k_global^6 / d2^3          # 排斥 (k/d)^6, 所有原子对都有
                is_edge && (V += d2 / k_global^2)   # 引力 (d/k)^2, 只有边
            end
        end
        return V
    end

    init_val = potential(x0)
    x_best = x0; best_val = init_val
    try
        r = optimize(potential, x0, LBFGS(), Optim.Options(iterations=actual_iters))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "reg3力场嵌入: LBFGS失败, 退回NelderMead" exception=e
    end
    try
        r = optimize(potential, x_best, NelderMead(), Optim.Options(iterations=actual_iters*2))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "reg3力场嵌入: NelderMead精修失败" exception=e
    end

    edge_dists = Float64[]
    nonedge_dists = Float64[]
    for i in 1:n_atoms, j in (i+1):n_atoms
        dx = x_best[2i-1]-x_best[2j-1]; dy = x_best[2i]-x_best[2j]
        d  = sqrt(dx^2 + dy^2)
        if (i,j) in edge_set
            push!(edge_dists, d)
        else
            push!(nonedge_dists, d)
        end
    end
    @printf("  [诊断] 力场嵌入(k_mode=%s) 势能: 初始=%.6g → 最终=%.6g\n", k_mode, init_val, best_val)
    @printf("  [诊断] 边距离(目标平衡点≈%.2f): min=%.2f  max=%.2f  mean=%.2f\n",
            d_target, minimum(edge_dists), maximum(edge_dists), mean(edge_dists))
    @printf("  [诊断] 非边距离: min=%.2f  max=%.2f  mean=%.2f\n",
            minimum(nonedge_dists), maximum(nonedge_dists), mean(nonedge_dists))

    positions = [(x_best[2i-1], x_best[2i], 0.0) for i in 1:n_atoms]
    return AtomList(positions), best_val
end

"""带权随机 3-regular 图, 用 embed_reg3_force_2d 的连续力场嵌入, 权重从收敛后
   的实际距离读出。k_mode 见 embed_reg3_force_2d 的文档字符串(global/local/
   legacy 三选一)。同时返回一个"unwanted interaction"诊断量: 非边对贡献的
   耦合总量 / 边对贡献的耦合总量, 越接近0说明分离得越干净。"""
function make_reg3_weighted_graph_force(n_atoms::Int, seed::Int=42;
                                         k_mode::String="legacy",
                                         d_target::Float64=12.0)
    edges = make_reg3_graph(n_atoms, seed)
    atoms, stress_val = embed_reg3_force_2d(edges, n_atoms;
                                             k_mode=k_mode, d_target=d_target, seed=seed)
    pos = [atoms[i] for i in 1:n_atoms]
    edge_set = Set((min(i,j),max(i,j)) for (i,j) in edges)
    weights = Float64[]
    edge_coupling_sum = 0.0
    nonedge_coupling_sum = 0.0
    for (i,j) in edges
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        w = C6_VAL/(4.0*r2^3)
        push!(weights, w)
        edge_coupling_sum += w
    end
    for i in 1:n_atoms, j in (i+1):n_atoms
        (i,j) in edge_set && continue
        xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
        r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
        nonedge_coupling_sum += C6_VAL/(4.0*r2^3)
    end
    unwanted_ratio = edge_coupling_sum > 0 ? nonedge_coupling_sum/edge_coupling_sum : NaN
    return atoms, edges, weights, stress_val, unwanted_ratio
end
end # @everywhere begin (geometric weighted graph)

@everywhere begin
"""由边权重 w_jk 反推"需要的相互作用强度" V_jk = 4*w_jk, 再反解出对应的
   原子间距离 r_jk = (C6/V_jk)^(1/6)。V_jk=4*w_jk 这个系数是把 Rydberg Ising
   Hamiltonian 的 σᶻσᶻ 项系数 (1/4)V_jk 跟目标 Hamiltonian Σw_jk σᶻσᶻ 的系数
   直接对齐得到的, 跟 Bauer et al. Eq.(11) 的 w_ij=C6/(4r^6) 是同一套物理关系。"""
function weighted_graph_target_distances(edges, weights::Vector{Float64}, C6::Float64)
    r_target = Dict{Tuple{Int,Int},Float64}()
    for (idx,(i,j)) in enumerate(edges)
        Vij = 4.0 * weights[idx]
        r_target[(min(i,j),max(i,j))] = (C6 / Vij)^(1/6)
    end
    return r_target
end

"""把目标距离嵌入到二维平面上的原子坐标。
   目标函数 (stress): 有边的原子对, 惩罚 (实际距离-目标距离)²；没有边的原子对,
   只有当距离小于一个"安全排斥距离" r_repel 时才惩罚 (单边弹簧, 不强迫它们
   飞到无穷远)。r_repel 取所有目标边距离里最大的那个乘一个安全系数, 用来压低
   非边原子对意外靠太近产生的额外耦合("unwanted interactions")。用 Optim 的
   LBFGS 求解 —— 这是个光滑的最小二乘型目标, 跟 QAOA 那种带采样噪声的量子
   期望值目标不一样, LBFGS 在这类图嵌入/MDS 问题上是标准选择, 应该比较稳;
   仍然包一层 try/catch, 失败就退到 NelderMead。"""
function embed_weighted_graph_2d(edges, weights::Vector{Float64}, n_atoms::Int;
                                  C6::Float64=C6_VAL, seed::Int=42,
                                  repel_factor::Float64=1.8,
                                  lbfgs_iters::Int=500, nm_iters::Int=2000)
    r_target = weighted_graph_target_distances(edges, weights, C6)
    r_max = maximum(values(r_target))
    r_repel = repel_factor * r_max
    edge_set = Set(keys(r_target))

    Random.seed!(seed + 2000)
    x0 = (rand(2*n_atoms) .- 0.5) .* (2.0 * r_max * sqrt(n_atoms))

    function stress(x::Vector{Float64})
        s = 0.0
        for i in 1:n_atoms, j in (i+1):n_atoms
            dx = x[2i-1]-x[2j-1]; dy = x[2i]-x[2j]
            d  = sqrt(dx^2 + dy^2 + 1e-12)
            key = (i,j)
            if key in edge_set
                s += (d - r_target[key])^2
            else
                pen = max(0.0, r_repel - d)
                s += pen^2
            end
        end
        return s
    end

    x_best = x0; best_val = stress(x0)
    try
        r = optimize(stress, x0, LBFGS(), Optim.Options(iterations=lbfgs_iters))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "图嵌入: LBFGS 失败, 退回 NelderMead" exception=e
    end
    try
        r = optimize(stress, x_best, NelderMead(), Optim.Options(iterations=nm_iters))
        v = Optim.minimum(r)
        v < best_val && (x_best = Optim.minimizer(r); best_val = v)
    catch e
        @warn "图嵌入: NelderMead 精修失败" exception=e
    end

    positions = [(x_best[2i-1], x_best[2i], 0.0) for i in 1:n_atoms]
    return AtomList(positions), best_val, r_target, r_repel
end

"""embed_weighted_graph_2d 的无权版本 —— 给所有边喂一个"统一权重"
   w_unit = C6_VAL/(4*d^6), 这样反解出来的目标距离对每条边都精确等于 d
   (代入验证: r=(C6/(4*w_unit))^(1/6) = (C6/(4*C6/(4d^6)))^(1/6) = d)。
   用于修复 q4/reg3/udisk 里边距离不一致的问题: 拓扑生成 (make_q4_edges /
   make_reg3_graph / 随机几何图的边判定) 保持不变, 只是把原来"硬编码位置公式"
   或"位置跟拓扑无关的网格摆放"换成这里的统一距离嵌入。"""
function embed_unweighted_graph_2d(edges, n_atoms::Int, d::Float64;
                                    seed::Int=42, repel_factor::Float64=1.8)
    n_edges = length(edges)
    w_unit  = C6_VAL / (4.0 * d^6)
    weights = fill(w_unit, n_edges)
    atoms, stress_val, _, _ = embed_weighted_graph_2d(
        edges, weights, n_atoms; C6=C6_VAL, seed=seed, repel_factor=repel_factor)
    return atoms, stress_val
end # @everywhere begin (weighted graph embedding)
end

# [FIX] maxcut_ev_weighted / loss_fn_warmstart / loss_fn_warmstart_weighted
# 原来定义在两个@everywhere代码块的空隙里, 只在主进程上有定义——这是跟
# make_reg3_weighted_graph同一类漏洞(见聊天记录), 之前没暴露过是因为
# gnn_qaoa_weighted从来没真正触发过并行worker。这次给 gnn_pretrain_and_
# qaoa_weighted 真正加了并行worker后, pmap把三个condition(包括cold_start)
# 的闭包一起发去worker——闭包里那句 `cnd=="gnn_warmstart" ? loss_fn_
# warmstart_weighted(...) : loss_fn_weighted(...)` 即使运行时走的是
# loss_fn_weighted那个分支, Julia序列化闭包时依然要能解析到
# loss_fn_warmstart_weighted这个引用, worker上找不到就直接报错——这也是
# 为什么连cold_start都失败了, 不是cold_start自己的逻辑有问题。
# maxcut_ev_weighted 同理: loss_fn_weighted 内部调用它, 它不是@everywhere
# 的话 loss_fn_weighted 在worker上也会解析失败。三个一起纳入@everywhere块。
@everywhere begin
"""maxcut_ev 的加权版本: 用末态的测量概率分布算期望的加权 cut 值
   C_obt = Σ_b P(b) * Σ_{(j,k)∈E} w_jk * 1[z_j≠z_k]。写法跟文件里已有的
   (无权) maxcut_ev 完全一致, 只是把"+1"换成"+w_jk"。"""
function maxcut_ev_weighted(reg::ArrayReg, edges, weights::Vector{Float64})
    sv = reg.state; val = 0.0
    for b in 0:(2^nqubits(reg)-1)
        prob = abs2(sv[b+1]); prob > 1e-12 || continue
        cut = 0.0
        for (idx,(u,v)) in enumerate(edges)
            zu = (b>>(u-1))&1; zv = (b>>(v-1))&1
            zu != zv && (cut += weights[idx])
        end
        val += prob * cut
    end
    return val
end

"""暴力枚举经典最优解 C_opt (2^n_atoms 种配置)。图规模建议 ≤ 22~24,
   否则会很慢 —— 随机 3-regular 图的测试建议先用 n_atoms ≤ 16。"""
function brute_force_maxcut(edges, weights::Vector{Float64}, n_atoms::Int)
    n_atoms > 24 && @warn "n_atoms=$(n_atoms) 偏大, 暴力枚举 (2^$(n_atoms)) 可能很慢"
    best = -Inf
    for b in 0:(2^n_atoms-1)
        cut = 0.0
        for (idx,(u,v)) in enumerate(edges)
            zu = (b>>(u-1))&1; zv = (b>>(v-1))&1
            zu != zv && (cut += weights[idx])
        end
        cut > best && (best = cut)
    end
    return best
end


"""Loss function using GNN-biased 初态制备 + 定制mixer (warm-start QAOA,
   Egger et al. 2021 完整方案, 也是 Jain et al. 2022 GNN warm-start 论文引用
   的来源) —— [FIX v3] 之前只做了"有偏初态制备", mixer部分还是跟cold_start
   一样的标准均匀mixer(Ω=Ωt对所有原子相同)——这不是Egger et al.完整方案。
   原文Eq.12给出的定制mixer哈密顿量是一个跟GNN概率pᵢ绑定的2×2矩阵:
     H_mixer,i = q_i·Z_i - 2√(p_i(1-p_i))·X_i,  q_i=2p_i-1
   pᵢ=0.5(完全不确定)时, q_i=0, X系数=2√(0.25)=1, 退化回标准mixer(跟原来
   一致); pᵢ→0或1(GNN很确信)时, X系数趋于0(几乎不再翻转这个原子), 同时
   多出一个正比于q_i的Z偏置——这才是Egger完整方案的意图: GNN越确信的原子,
   mixer越"放过"它, 只在mixer段体现, 不影响cost段和态制备段。
   在我们的Rydberg框架里实现: mixer段的Ω从"全原子统一Ωt"换成逐原子的
   Ωt·2√(pᵢ(1-pᵢ))(pᵢ=0.5时数值=1, 精确对应Ωt, 跟原来完全一致); 同时mixer
   段(且仅在mixer段)的失谐场在原有Δᵢ基础上额外叠加 Ωt·qᵢ 这一项(用跟X项
   同量级的Ωt做归一化, 不引入新的自由参数)。cost段/态制备段的Δᵢ、Ω均不变。
   Δ 仍然用跟 loss_fn 一样的论文精确公式(对所有原子对求和); 态制备仍然是
   逐原子 θᵢ=2·arcsin(√pᵢ) 的旋转。
   graph_type 参数保留只是为了兼容旧调用签名, 现在不再区分处理。"""
function loss_fn_warmstart(x::AbstractVector{T},
                            p_gnn::Vector{Float64},
                            atoms, edges, n_atoms;
                            graph_type::String="q3") where T
    @assert length(x) % 2 == 0
    p_cur = length(x) ÷ 2
    Ω0=5π; t0=π/(2*Ω0); Ωt=OMEGA_T_REF[]; C6=2π*862690

    # Δ: 跟 loss_fn 完全一样的论文精确公式, 对所有原子对求和
    pos = [atoms[i] for i in 1:n_atoms]
    sw = zeros(n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        xi,yi,zi=pos[i]; xj,yj,zj=pos[j]
        r2=(xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        sw[i] += C6/(4*r2^3)
    end
    Δt = 2.0*sw

    # 态制备: 逐原子旋转角度 θᵢ=2·arcsin(√pᵢ), 固定时长t0内转完 → 逐原子Ω
    p_clamped = clamp.(p_gnn, max(1e-6, WS_EPS_G), 1.0-max(1e-6, WS_EPS_G))   # WS_EPS_G=0 ⇒ 与原来逐位一致
    θ_prep = 2.0 .* asin.(sqrt.(p_clamped))
    Ω_prep = θ_prep ./ t0

    # [FIX v3] Egger et al. Eq.12 定制mixer: q_i·Z_i - 2√(p_i(1-p_i))·X_i
    q = 2.0 .* p_clamped .- 1.0
    Ω_mix_coef = 2.0 .* sqrt.(p_clamped .* (1.0 .- p_clamped))   # p=0.5时=1, 退化回标准mixer

    dur = vcat(t0, abs.(x))
    clk = [0.0, cumsum(dur)...]

    Ωwf = Vector{Any}(undef, n_atoms)
    Δwf = Vector{Any}(undef, n_atoms)
    ϕwf = Vector{Any}(undef, n_atoms)
    for i in 1:n_atoms
        Ωvals = Float64[Ω_prep[i]]   # 态制备段
        Δvals = Float64[Δt[i]]
        for layer in 1:p_cur
            push!(Ωvals, 0.0);              push!(Δvals, Δt[i])                  # cost段: 不变
            push!(Ωvals, Ωt*Ω_mix_coef[i]); push!(Δvals, Δt[i] + WS_MIXSIGN_G*Ωt*q[i])   # mixer段: 定制(符号见 WS_MIXSIGN_G)
        end
        Ωwf[i] = piecewise_constant(clocks=clk, values=Ωvals)
        Δwf[i] = piecewise_constant(clocks=clk, values=Δvals)
        ϕwf[i] = piecewise_constant(clocks=clk, values=vcat(-0.5π, zeros(T, 2*p_cur)))
    end

    h    = rydberg_h(atoms; Δ=Δwf, Ω=Ωwf, ϕ=ϕwf)
    reg  = zero_state(Complex{T}, n_atoms)
    prob = KrylovEvolution(reg, clk, h); emulate!(prob)
    return -maxcut_ev(prob.reg, edges)
end

"""loss_fn_warmstart 的带权版本, 同样的"有偏初态制备+定制mixer"逻辑(见
   loss_fn_warmstart 文档字符串), 只是最后的测量/优化目标换成带权版本。"""
function loss_fn_warmstart_weighted(x::AbstractVector{T}, p_gnn::Vector{Float64},
                                     atoms, edges, weights::Vector{Float64}, n_atoms;
                                     mixsign::Float64=WS_MIXSIGN_G) where T
    @assert length(x) % 2 == 0
    p_cur = length(x) ÷ 2
    Ω0=5π; t0=π/(2*Ω0); Ωt=OMEGA_T_REF[]; C6=2π*862690

    pos = [atoms[i] for i in 1:n_atoms]
    sw = zeros(n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        xi,yi,zi=pos[i]; xj,yj,zj=pos[j]
        r2=(xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        sw[i] += C6/(4*r2^3)
    end
    Δt = 2.0*sw

    p_clamped = clamp.(p_gnn, max(1e-6, WS_EPS_G), 1.0-max(1e-6, WS_EPS_G))   # WS_EPS_G=0 ⇒ 与原来逐位一致
    θ_prep = 2.0 .* asin.(sqrt.(p_clamped))
    Ω_prep = θ_prep ./ t0

    # [FIX v3] Egger et al. Eq.12 定制mixer, 跟 loss_fn_warmstart 同样的做法
    q = 2.0 .* p_clamped .- 1.0
    Ω_mix_coef = 2.0 .* sqrt.(p_clamped .* (1.0 .- p_clamped))

    dur = vcat(t0, abs.(x))
    clk = [0.0, cumsum(dur)...]

    Ωwf = Vector{Any}(undef, n_atoms)
    Δwf = Vector{Any}(undef, n_atoms)
    ϕwf = Vector{Any}(undef, n_atoms)
    for i in 1:n_atoms
        Ωvals = Float64[Ω_prep[i]]
        Δvals = Float64[Δt[i]]
        for layer in 1:p_cur
            push!(Ωvals, 0.0);              push!(Δvals, Δt[i])
            # mixsign = +1: 原实现(所有已有结果); mixsign = −1: Egger 精确版 —— 按 Bloqade 约定, 只有失谐取
            #   −Ωt·q_i 时初态才是定制 mixer 的本征态(+1 时 Z 分量反号, 初态会被 mixer 转动)
            push!(Ωvals, Ωt*Ω_mix_coef[i]); push!(Δvals, Δt[i] + mixsign*Ωt*q[i])
        end
        Ωwf[i] = piecewise_constant(clocks=clk, values=Ωvals)
        Δwf[i] = piecewise_constant(clocks=clk, values=Δvals)
        ϕwf[i] = piecewise_constant(clocks=clk, values=vcat(-0.5π, zeros(T, 2*p_cur)))
    end

    h    = rydberg_h(atoms; Δ=Δwf, Ω=Ωwf, ϕ=ϕwf)
    reg  = zero_state(Complex{T}, n_atoms)
    prob = KrylovEvolution(reg, clk, h); emulate!(prob)
    return -maxcut_ev_weighted(prob.reg, edges, weights)
end

"""[B 方案] GNN 偏置场退火 (gnn_bias): 冷启动的退火结构 + 由 GNN 决定、逐层衰减的局部纵向偏置场。
   思路来自"带纵向偏置场的量子退火"(用经典/平均场解作偏置, 偏置在退火过程中撤掉)。
   与 Egger 预热(loss_fn_warmstart_weighted)的区别:
     · 混合层的横场始终是全幅 Ωt(所有原子相同, 与冷启动一样), 不随 GNN 置信度削弱 ⇒ 没有原子被冻结;
     · 混合层额外的失谐 −λ_k·Ωt·q_i(q_i = 2p_i − 1)只是"引导"。符号按 Bloqade 约定
       H = Ω/2(e^{iϕ}|g⟩⟨r| + h.c.) − Δn 推出: 态制备(ϕ=−π/2)给出 Bloch 矢量 (2√(p(1−p)), 0, 1−2p),
       它平行于混合层场 (Ωt, 0, −λΩt q)/2 ⇔ 失谐取 −λΩt q。这与冷启动 |+⟩ 是 +x̂ 场的本征态属同一分支。
       λ_k = λ0·r^(k−1) 逐层衰减 ⇒ 浅层像 GNN 预热, 深层回到冷启动的退火分支(终点哈密顿量不含偏置);
     · 代价层完全不变(Ω=0, Δ_i 只抵消线性项), 所以目标函数仍是原 MaxCut。
   初态: prep=true 时取第 1 层混合层的本征态 —— 等价于用"软化"概率 p_eff 的预热初态,
     2p_eff − 1 = λ0·q_i / √(1 + λ0²q_i²)(λ0→0 退化为冷启动 |+⟩, λ0→∞ 趋于 GNN 硬判决);
     prep=false 时与冷启动完全相同(全局 Ω0 的 π/2 脉冲)。
   λ0 = 0 时与 loss_fn_weighted 逐位一致(可作核查)。"""
function loss_fn_bias_weighted(x::AbstractVector{T}, p_gnn::Vector{Float64}, atoms, edges,
                               weights::Vector{Float64}, n_atoms; λ0::Float64=1.0, r::Float64=0.7,
                               prep::Bool=true, mode::Symbol=:geo) where T
    @assert length(x) % 2 == 0
    p_cur = length(x) ÷ 2
    Ω0=5π; t0=π/(2*Ω0); Ωt=OMEGA_T_REF[]; C6=2π*862690
    pos = [atoms[i] for i in 1:n_atoms]
    sw = zeros(n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        xi,yi,zi=pos[i]; xj,yj,zj=pos[j]
        r2=(xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        sw[i] += C6/(4*r2^3)
    end
    Δt = 2.0*sw
    pc = clamp.(p_gnn, 1e-6, 1.0-1e-6)
    q  = 2.0 .* pc .- 1.0
    a  = 2.0 .* sqrt.(pc .* (1.0 .- pc))
    if mode in (:interp, :egeo)                              # B2/B3: 初态 = 完整的 GNN 预热态(Egger 精确版的本征态)
        Ω_prep = 2.0 .* asin.(sqrt.(pc)) ./ t0
    elseif prep && λ0 > 0
        b    = λ0 .* q ./ sqrt.(1.0 .+ (λ0 .* q).^2)          # = 2p_eff − 1
        peff = clamp.((1.0 .+ b) ./ 2.0, 1e-6, 1.0-1e-6)
        Ω_prep = 2.0 .* asin.(sqrt.(peff)) ./ t0               # p_eff = 0.5 ⇒ Ω0(冷启动)
    else
        Ω_prep = fill(Ω0, n_atoms)
    end
    dur = vcat(t0, abs.(x))
    clk = [0.0, cumsum(dur)...]
    Ωwf = Vector{Any}(undef, n_atoms)
    Δwf = Vector{Any}(undef, n_atoms)
    ϕwf = Vector{Any}(undef, n_atoms)
    for i in 1:n_atoms
        Ωvals = Float64[Ω_prep[i]]
        Δvals = Float64[Δt[i]]
        for layer in 1:p_cur
            push!(Ωvals, 0.0); push!(Δvals, Δt[i])                     # 代价层: 不变
            if mode in (:interp, :egeo)
                # B2 (interp): 混合层从 Egger 精确版定制 mixer (a_i X − q_i Z) 线性过渡到标准 mixer (X),
                #     w 按相对位置 s=(k−1)/(p−1) 取值 —— 混合层随总深度变, 与逐层贪心不兼容(实测会退化)。
                # B3 (egeo): w = 1 − r^(k−1) 只依赖层号 ⇒ 加层时前缀的哈密顿量不变, 与逐层 BO + 锁前缀兼容。
                w = mode == :egeo ? 1.0 - r^(layer - 1) : (p_cur == 1 ? 0.0 : (layer - 1) / (p_cur - 1))
                push!(Ωvals, Ωt*((1.0 - w)*a[i] + w)); push!(Δvals, Δt[i] - (1.0 - w)*Ωt*q[i])
            else
                # geo: λ_k = λ0·r^(k−1)(按层号衰减, B1); lin: λ_k = λ0·(1 − (k−1)/p)(按相对位置衰减)
                λk = mode == :lin ? λ0 * (1.0 - (layer - 1) / p_cur) : λ0 * r^(layer - 1)
                push!(Ωvals, Ωt);  push!(Δvals, Δt[i] - λk*Ωt*q[i])    # 混合层: 全幅横场 + 衰减的 GNN 偏置
            end
        end
        Ωwf[i] = piecewise_constant(clocks=clk, values=Ωvals)
        Δwf[i] = piecewise_constant(clocks=clk, values=Δvals)
        ϕwf[i] = piecewise_constant(clocks=clk, values=vcat(-0.5π, zeros(T, 2*p_cur)))
    end
    h    = rydberg_h(atoms; Δ=Δwf, Ω=Ωwf, ϕ=ϕwf)
    reg  = zero_state(Complex{T}, n_atoms)
    prob = KrylovEvolution(reg, clk, h); emulate!(prob)
    return -maxcut_ev_weighted(prob.reg, edges, weights)
end

# ── [QNG] 末态波函数: 量子自然梯度需要 |ψ(θ)⟩ 本身, 而不只是能量 ─────────────
# 下面两个函数跟 loss_fn_weighted / loss_fn_warmstart_weighted **逐行一致**,
# 唯一区别是返回演化后的寄存器(prob.reg)而不是 -<C>。刻意复制而不是重构原
# 函数: 保证 NelderMead 路径(以及已经跑完的集群 checkpoint)一个比特都不变。
# 一致性关系: -maxcut_ev_weighted(state_xxx(x,...), edges, weights) == loss_fn_xxx(x,...)
function state_weighted(x::AbstractVector{T}, atoms, n_atoms) where T
    @assert length(x) % 2 == 0
    p_cur = length(x) ÷ 2
    Ω0=5π; t0=π/(2*Ω0); Ωt=OMEGA_T_REF[]; C6=2π*862690
    pos = [atoms[i] for i in 1:n_atoms]
    sw = zeros(n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        xi,yi,zi=pos[i]; xj,yj,zj=pos[j]
        r2=(xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        sw[i] += C6/(4*r2^3)
    end
    Δt  = 2.0*sw
    dur = vcat(t0, abs.(x))
    clk = [0.0, cumsum(dur)...]
    Ωv  = vcat(Ω0, [repeat([0.0, Ωt], p_cur)...])
    ϕv  = vcat(-0.5π, zeros(T, 2 * p_cur))
    td  = clk[end]
    Δwf = [piecewise_constant(clocks=[0.0,td], values=[Δt[i]]) for i in 1:n_atoms]
    h   = rydberg_h(atoms; Δ=Δwf,
                    Ω=piecewise_constant(clocks=clk, values=Ωv),
                    ϕ=piecewise_constant(clocks=clk, values=ϕv))
    reg  = zero_state(Complex{T}, n_atoms)
    prob = KrylovEvolution(reg, clk, h); emulate!(prob)
    return prob.reg
end

function state_warmstart_weighted(x::AbstractVector{T}, p_gnn::Vector{Float64},
                                  atoms, n_atoms) where T
    @assert length(x) % 2 == 0
    p_cur = length(x) ÷ 2
    Ω0=5π; t0=π/(2*Ω0); Ωt=OMEGA_T_REF[]; C6=2π*862690
    pos = [atoms[i] for i in 1:n_atoms]
    sw = zeros(n_atoms)
    for i in 1:n_atoms, j in 1:n_atoms
        i == j && continue
        xi,yi,zi=pos[i]; xj,yj,zj=pos[j]
        r2=(xi-xj)^2+(yi-yj)^2+(zi-zj)^2
        sw[i] += C6/(4*r2^3)
    end
    Δt = 2.0*sw
    p_clamped = clamp.(p_gnn, max(1e-6, WS_EPS_G), 1.0-max(1e-6, WS_EPS_G))   # WS_EPS_G=0 ⇒ 与原来逐位一致
    θ_prep = 2.0 .* asin.(sqrt.(p_clamped))
    Ω_prep = θ_prep ./ t0
    q = 2.0 .* p_clamped .- 1.0
    Ω_mix_coef = 2.0 .* sqrt.(p_clamped .* (1.0 .- p_clamped))
    dur = vcat(t0, abs.(x))
    clk = [0.0, cumsum(dur)...]
    Ωwf = Vector{Any}(undef, n_atoms)
    Δwf = Vector{Any}(undef, n_atoms)
    ϕwf = Vector{Any}(undef, n_atoms)
    for i in 1:n_atoms
        Ωvals = Float64[Ω_prep[i]]
        Δvals = Float64[Δt[i]]
        for layer in 1:p_cur
            push!(Ωvals, 0.0);              push!(Δvals, Δt[i])
            push!(Ωvals, Ωt*Ω_mix_coef[i]); push!(Δvals, Δt[i] + WS_MIXSIGN_G*Ωt*q[i])
        end
        Ωwf[i] = piecewise_constant(clocks=clk, values=Ωvals)
        Δwf[i] = piecewise_constant(clocks=clk, values=Δvals)
        ϕwf[i] = piecewise_constant(clocks=clk, values=vcat(-0.5π, zeros(T, 2*p_cur)))
    end
    h    = rydberg_h(atoms; Δ=Δwf, Ω=Ωwf, ϕ=ϕwf)
    reg  = zero_state(Complex{T}, n_atoms)
    prob = KrylovEvolution(reg, clk, h); emulate!(prob)
    return prob.reg
end
end # @everywhere begin (maxcut_ev_weighted / loss_fn_warmstart / loss_fn_warmstart_weighted / state_*)

# ── GNN TRAINING MODE ─────────────────────────────────────────────────────────
# julia Q3_sge.jl gnn_train <graph_type> <n_atoms> [n_epochs] [d_fixed]
if length(ARGS) >= 1 && ARGS[1] == "gnn_train"
    graph_type = length(ARGS) >= 2 ? ARGS[2] : "q3"
    n_at       = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    n_epochs   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 300
    d_fixed    = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 14.0
    using Statistics, LinearAlgebra

    println("\n" * "="^65)
    println("GNN TRAINING: graph=$(graph_type)  n=$(n_at)  epochs=$(n_epochs)")
    println("="^65)

    atoms, edges, n_atoms, n_edges, label = make_graph(graph_type, n_at, d_fixed)
    println("Graph: $(label)  |V|=$(n_atoms)  |E|=$(n_edges)")

    # [诊断] 随机基线 + 暴力枚举真实最优, 一次性把标尺立好, 不用再猜 GNN 结果
    # 到底是好是差。
    rand_cuts = Float64[]
    for _ in 1:200
        x = rand(0:1, n_atoms)
        cut = 0
        for (u,v) in edges
            x[u] != x[v] && (cut += 1)
        end
        push!(rand_cuts, cut / n_edges)
    end
    @printf("[诊断] 随机基线 (200次平均随机划分): ar=%.4f\n", mean(rand_cuts))
    if n_atoms <= 20
        let best_cut_bf = 0
            for bits in 0:(2^n_atoms - 1)
                cut = 0
                for (u,v) in edges
                    ((bits>>(u-1))&1) != ((bits>>(v-1))&1) && (cut += 1)
                end
                cut > best_cut_bf && (best_cut_bf = cut)
            end
            @printf("[诊断] 暴力枚举真实最优: ar=%.4f (%d/%d 条边)\n",
                    best_cut_bf/n_edges, best_cut_bf, n_edges)
        end
    else
        println("[诊断] n_atoms > 20, 跳过暴力枚举 (太慢)")
    end

    # Build graph matrices
    A, D_mat, Ajs  = build_power_adj(n_atoms, edges, 3)
    BL, D_L, U, S  = build_line_graph(n_atoms, edges)
    L_G            = graph_laplacian(n_atoms, edges)

    # GNN initialisation
    local params
    params = init_lgnn(3, 32, 8)

    # Training with Adam-like updates (learning rate schedule)
    local best_cut, best_p
    lr = 0.005; best_cut = 0.0; best_p = fill(0.5, n_atoms)
    println("\nTraining LGNN on $(label)...")

    eval_fn(p) = gnn_loss(
        lgnn_forward(p, A, D_mat, Ajs, BL, D_L, U, S, n_atoms, length(edges)), L_G)

    # [新增] 离散cut值单独跟踪, 不要只看连续松弛的ar —— pᵢ 不严格等于0/1时,
    # 连续公式会低估真实的离散cut, 跟随机基线/暴力枚举(都是离散量)不是同一个
    # 量纲, 直接比较会得出误导性的结论。
    local best_discrete, best_discrete_p
    best_discrete = 0.0; best_discrete_p = fill(0.5, n_atoms)
    discrete_cut(pvec) = begin
        cut = 0
        for (u,v) in edges
            (pvec[u] > 0.5) != (pvec[v] > 0.5) && (cut += 1)
        end
        cut / n_edges
    end

    for epoch in 1:n_epochs
        # Forward pass
        p = lgnn_forward(params, A, D_mat, Ajs, BL, D_L, U, S, n_atoms, length(edges))
        loss = gnn_loss(p, L_G)
        ar = -loss / n_edges   # approximation ratio from GNN alone (连续松弛)
        if ar > best_cut
            best_cut = ar; best_p = copy(p)
        end
        dar = discrete_cut(p)   # [新增] 把当前p按0.5取整后的真实离散cut
        if dar > best_discrete
            best_discrete = dar; best_discrete_p = copy(p)
        end

        # SPSA gradient estimate + update, over ALL parameter blocks [FIX: 原来
        # 只扰动 theta[1], 现在同时扰动 theta/phi/W_out/b_out 全部, 仍然只用
        # 2次额外前向计算 (spsa_update_all 内部实现)。
        dk = 0.05 / (epoch^0.101)   # perturbation scale
        ak = lr / (epoch^0.602)
        params = spsa_update_all(params, dk, ak, eval_fn)

        epoch % 50 == 0 &&
            @printf("  epoch %4d  GNN loss=%.4f  ar(连续)=%.4f  best(连续)=%.4f  best(离散)=%.4f\n",
                    epoch, loss, ar, best_cut, best_discrete)
    end

    println("\nGNN training complete:")
    @printf("  Best GNN approximation ratio (连续松弛, 仅供参考): %.4f\n", best_cut)
    @printf("  Best GNN approximation ratio (真实离散cut, 应该拿这个跟随机基线/暴力枚举比): %.4f\n", best_discrete)
    @printf("  Expected cut (离散): %.2f / %d\n", best_discrete * n_edges, n_edges)
    println("  GNN probabilities p_i (按离散最优结果对应的p, 划分):")
    for i in 1:n_atoms
        partition = best_discrete_p[i] > 0.5 ? "A" : "B"
        @printf("    atom %2d: p=%.3f → partition %s\n", i, best_discrete_p[i], partition)
    end

    # Save GNN params
    mkpath(directoryBasis)
    tag = "gnn_$(graph_type)_n$(n_atoms)"
    serialize(joinpath(directoryBasis, "$(tag).jls"), (params, best_discrete_p, best_discrete, n_atoms, n_edges, edges))
    println("✓ Saved → " * joinpath(directoryBasis, tag * ".jls"))
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── gnn_train_weighted: 跟 gnn_train 逐行对应的加权版本 ────────────────────
# 同样是纯经典训练 (lgnn_forward/gnn_loss 都是矩阵运算, 不碰 Bloqade/Rydberg
# 时间演化), 只有两处不同: (1) 图是带权随机3-regular (make_reg3_weighted_graph),
# (2) 用 weighted_graph_laplacian 而不是 graph_laplacian, ar 归一化用 Σw 而不是
# 边数。缓存文件名 gnn_reg3w_n<n_atoms>_seed<seed>.jls 跟 gnn_qaoa_weighted /
# gdqc_validate_weighted 内部查找缓存用的是同一套规则, 跑完这个之后跑那两个
# 模式会直接读到, 不用重新训练。
#
# 运行方式:
#   julia r3g_sge.jl gnn_train_weighted [n_atoms] [n_epochs] [seed]
# 例如:
#   julia r3g_sge.jl gnn_train_weighted 10 300 42
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "gnn_train_weighted"
    n_at     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    n_epochs = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 300
    seed     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 42
    using Statistics, LinearAlgebra

    println("\n" * "="^65)
    println("GNN TRAINING (加权): graph=reg3w  n=$(n_at)  epochs=$(n_epochs)  seed=$(seed)")
    println("="^65)

    edges, weights = make_reg3_weighted_graph(n_at, seed; wmin=0.5, wmax=1.0)
    n_atoms = n_at; n_edges = length(edges)
    label   = "reg3-weighted(n=$(n_atoms),seed=$(seed))"
    println("Graph: $(label)  |V|=$(n_atoms)  |E|=$(n_edges)")
    @printf("权重范围: [%.3f, %.3f]  Σw=%.3f\n", minimum(weights), maximum(weights), sum(weights))

    # Build graph matrices — 唯一跟无权版不同的地方: 加权拉普拉斯矩阵
    A, D_mat, Ajs  = build_power_adj(n_atoms, edges, 3)
    BL, D_L, U, S  = build_line_graph(n_atoms, edges)
    L_G            = weighted_graph_laplacian(n_atoms, edges, weights)
    total_w        = sum(weights)
    node_wd        = weighted_node_degree(n_atoms, edges, weights)   # [新增] 加权度数
    @printf("加权度数范围: [%.3f, %.3f]  (拓扑度数全部=3, 加权度数天然打破对称性)\n",
            minimum(node_wd), maximum(node_wd))

    # [诊断] 随机基线 + 暴力枚举真实最优(加权版), 跟无权版 gnn_train 一样先把
    # 标尺立好。
    rand_cuts_w = Float64[]
    for _ in 1:200
        x = rand(0:1, n_atoms)
        cutw = 0.0
        for (idx,(u,v)) in enumerate(edges)
            x[u] != x[v] && (cutw += weights[idx])
        end
        push!(rand_cuts_w, cutw / total_w)
    end
    @printf("[诊断] 随机基线 (200次平均随机划分, 加权): ar=%.4f\n", mean(rand_cuts_w))
    if n_atoms <= 20
        let best_cut_bf = 0.0
            for bits in 0:(2^n_atoms - 1)
                cutw = 0.0
                for (idx,(u,v)) in enumerate(edges)
                    ((bits>>(u-1))&1) != ((bits>>(v-1))&1) && (cutw += weights[idx])
                end
                cutw > best_cut_bf && (best_cut_bf = cutw)
            end
            @printf("[诊断] 暴力枚举真实最优 (加权): ar=%.4f (权重 %.3f/%.3f)\n",
                    best_cut_bf/total_w, best_cut_bf, total_w)
        end
    else
        println("[诊断] n_atoms > 20, 跳过暴力枚举 (太慢)")
    end

    local params
    params = init_lgnn(3, 32, 8)

    local best_cut, best_p
    lr = 0.005; best_cut = 0.0; best_p = fill(0.5, n_atoms)
    println("\nTraining LGNN on $(label) (weighted QUBO loss)...")

    # [新增] 离散cut(加权)单独跟踪 —— 连续松弛的ar会低估真实离散cut, 跟随机
    # 基线/暴力枚举(都是离散量)不是同一个量纲, 直接比较会得出误导性结论。
    local best_discrete, best_discrete_p
    best_discrete = 0.0; best_discrete_p = fill(0.5, n_atoms)
    discrete_cut_w(pvec) = begin
        cutw = 0.0
        for (idx,(u,v)) in enumerate(edges)
            (pvec[u] > 0.5) != (pvec[v] > 0.5) && (cutw += weights[idx])
        end
        cutw / total_w
    end

    eval_fn(p) = gnn_loss(
        lgnn_forward(p, A, D_mat, Ajs, BL, D_L, U, S, n_atoms, length(edges);
                     node_feat1=node_wd, edge_feat1=weights), L_G)

    for epoch in 1:n_epochs
        p = lgnn_forward(params, A, D_mat, Ajs, BL, D_L, U, S, n_atoms, length(edges);
                          node_feat1=node_wd, edge_feat1=weights)
        loss = gnn_loss(p, L_G)
        ar = -loss / total_w   # 加权版 approximation ratio (连续松弛): cut权重/总权重
        if ar > best_cut
            best_cut = ar; best_p = copy(p)
        end
        dar = discrete_cut_w(p)   # [新增] 按0.5取整后的真实离散cut(加权)
        if dar > best_discrete
            best_discrete = dar; best_discrete_p = copy(p)
        end

        # SPSA梯度估计+更新, 同时扰动全部参数块 [FIX: 原来只扰动 theta[1]]
        dk = 0.05 / (epoch^0.101)
        ak = lr / (epoch^0.602)
        params = spsa_update_all(params, dk, ak, eval_fn)

        epoch % 50 == 0 &&
            @printf("  epoch %4d  GNN loss=%.4f  ar(连续)=%.4f  best(连续)=%.4f  best(离散)=%.4f\n",
                    epoch, loss, ar, best_cut, best_discrete)
    end

    println("\nGNN training complete (加权):")
    @printf("  Best GNN approximation ratio (连续松弛, 仅供参考): %.4f\n", best_cut)
    @printf("  Best GNN approximation ratio (真实离散cut, 应该拿这个跟随机基线/暴力枚举比): %.4f\n", best_discrete)
    @printf("  Expected cut weight (离散): %.3f / %.3f\n", best_discrete * total_w, total_w)
    println("  GNN probabilities p_i (按离散最优结果对应的p, 划分):")
    for i in 1:n_atoms
        partition = best_discrete_p[i] > 0.5 ? "A" : "B"
        @printf("    atom %2d: p=%.3f → partition %s\n", i, best_discrete_p[i], partition)
    end

    mkpath(directoryBasis)
    tag = "gnn_reg3w_n$(n_atoms)_seed$(seed)"
    serialize(joinpath(directoryBasis, "$(tag).jls"), (params, best_discrete_p, best_discrete, n_atoms, n_edges, edges))
    println("✓ Saved → " * joinpath(directoryBasis, tag * ".jls"))
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── gnn_pretrain_weighted: GNN 元学习式预训练 (对应 Jain et al. 2022 Eq.15) ──
# 跟 gnn_train_weighted 的本质区别: 那个是对*一张具体的图*做优化搜索(参数
# 只对那一张图有意义); 这个是在*一批不同的图*(混合不同n_atoms、不同seed)
# 上训练, 代价函数是这批图各自代价的平均值(Eq.15), 训出来的参数是一个
# 可迁移的"先验", 存好之后可以直接对任意新图做一次前向推理, 不用重新训练。
# 训练池默认覆盖 n_atoms ∈ [n_min, n_max] (间隔 n_step), 每个规模 seeds_per_size
# 个不同种子的图。
#
# 运行方式:
#   julia r3g_sge.jl gnn_pretrain_weighted <n_epochs> [batch_size] [n_min] [n_max] [n_step] [seeds_per_size]
# 例如:
#   julia r3g_sge.jl gnn_pretrain_weighted 2000 8 8 20 2 20
#   (2000轮, 每轮从训练池抽8张图, n_atoms覆盖8,10,12,...,20, 每个规模20个种子,
#    训练池总共 7*20=140 张图)
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "gnn_pretrain_weighted"
    n_epochs        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2000
    batch_size      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    n_min           = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8
    n_max           = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20
    n_step          = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2
    seeds_per_size  = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 20
    using Statistics, LinearAlgebra

    n_list = collect(n_min:n_step:n_max)
    println("\n" * "="^70)
    println("GNN 预训练 (加权, 对应论文Eq.15批量平均loss)")
    println("n_atoms 范围: $(n_list)  ·  每个规模 $(seeds_per_size) 个种子  ·  epochs=$(n_epochs)  ·  batch_size=$(batch_size)")
    println("="^70)

    pool = build_graph_pool_weighted(n_list, seeds_per_size)
    println("训练池共 $(length(pool)) 张图\n")

    params, best_avg = train_lgnn_pretrain_weighted(pool; n_epochs=n_epochs, batch_size=batch_size)

    mkpath(directoryBasis)
    tag = "gnn_prior_weighted_n$(n_min)-$(n_max)_seeds$(seeds_per_size)"
    serialize(joinpath(directoryBasis, "$(tag).jls"), params)
    @printf("\n✓ 预训练完成(保存的是训练过程中历史最优, 不是训练结束时的参数), 训练池平均真实离散cut=%.4f\n", best_avg)
    println("  先验参数已保存 → " * joinpath(directoryBasis, tag * ".jls"))
    println("  这份参数不跟任何具体图绑定, 可以用 gnn_qaoa_weighted 的 use_prior 选项")
    println("  直接对任意新图做一次推理, 不用重新训练。")
    println("="^70)
    exit(0)
end

# ── GNN+QAOA COMBINED MODE ────────────────────────────────────────────────────
# julia Q3_sge.jl gnn_qaoa <graph_type> <n_atoms> <p_val> [d_fixed]
# 1. Loads or trains GNN on the specified graph
# 2. Uses GNN probabilities as hardware-native warm-start
# 3. Runs GSM+CSM BO on the warm-started QAOA
# Compares: (a) cold-start BO  (b) TQA-initialised BO  (c) GNN warm-start BO
if length(ARGS) >= 1 && ARGS[1] == "gnn_qaoa"
    # ── GNN_QAOA CHAIN ──────────────────────────────────────────────────────
    # julia r3g_sge.jl gnn_qaoa <graph_type> <n_atoms> [d_fixed] [p_max]
    #
    # Runs p=1→p_max comparing THREE initialisation conditions on the same graph:
    #   1. cold_start    — random Halton (our baseline)
    #   2. tqa_init      — Trotterised Quantum Annealing (δt=0.75, Sack & Serbyn 2021)
    #   3. gnn_warmstart — LGNN probabilities → Rydberg hardware-native detuning
    #
    # All conditions use identical GSM+CSM BO, identical budget, identical seeds.
    # Recommended: julia r3g_sge.jl gnn_qaoa q3 8 14.0 5

    graph_type = length(ARGS) >= 2 ? ARGS[2] : "q3"
    n_at       = length(ARGS) >= 3 ? parse(Int,     ARGS[3]) : 8
    d_fixed    = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 14.0
    p_max      = length(ARGS) >= 5 ? parse(Int,     ARGS[5]) : 5
    using Statistics, LinearAlgebra

    atoms, edges, n_atoms, n_edges, label = make_graph(graph_type, n_at, d_fixed; seed=42)

    println("\n" * "="^70)
    println("INITIALISATION COMPARISON: cold-start vs TQA vs GNN warm-start")
    println("Graph:  $(label)  |V|=$(n_atoms)  |E|=$(n_edges)")
    println("d:      $(d_fixed) μm (fixed optimal)")
    println("p:      1 → $(p_max)  ·  3 conditions  ·  $(length(SEEDS)) seeds each")
    println("="^70)

    # ── Step 1: Train LGNN (or load cached) ─────────────────────────────────
    gnn_file = joinpath(directoryBasis, "gnn_$(graph_type)_n$(n_atoms).jls")
    local gnn_params, best_p_gnn, gnn_ar

    if isfile(gnn_file)
        println("\nLoading pre-trained LGNN from $(basename(gnn_file))")
        gnn_params, best_p_gnn, gnn_ar, _, _, _ = deserialize(gnn_file)
        @printf("  LGNN standalone ar = %.4f\n", gnn_ar)
    else
        println("\nTraining LGNN (300 epochs, unsupervised MaxCut QUBO)...")
        A_lg, D_lg, Ajs_lg = build_power_adj(n_atoms, edges, 3)
        BL_lg, DL_lg, U_lg, S_lg = build_line_graph(n_atoms, edges)
        L_G = graph_laplacian(n_atoms, edges)
        gnn_params = init_lgnn(3, 32, 8)
        best_p_gnn = fill(0.5, n_atoms); gnn_ar = 0.0
        for epoch in 1:200
            p_out = lgnn_forward(gnn_params, A_lg, D_lg, Ajs_lg,
                                 BL_lg, DL_lg, U_lg, S_lg, n_atoms, length(edges))
            ar_e  = -gnn_loss(p_out, L_G) / n_edges
            ar_e > gnn_ar && (gnn_ar = ar_e; best_p_gnn = copy(p_out))
            dk = 0.05/(epoch^0.101)
            delta = randn(size(gnn_params.theta[1])) .* dk
            ak = 0.005/(epoch^0.602)
            tp  = deepcopy(gnn_params.theta); tp[1]  += delta
            tm  = deepcopy(gnn_params.theta); tm[1]  -= delta
            pp_ = LGNNParams(tp, gnn_params.phi, gnn_params.J, gnn_params.feat_dim,
                             gnn_params.T_iter, gnn_params.W_out, gnn_params.b_out)
            pm_ = LGNNParams(tm, gnn_params.phi, gnn_params.J, gnn_params.feat_dim,
                             gnn_params.T_iter, gnn_params.W_out, gnn_params.b_out)
            pp_out = lgnn_forward(pp_, A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg, n_atoms, length(edges))
            pm_out = lgnn_forward(pm_, A_lg, D_lg, Ajs_lg, BL_lg, DL_lg, U_lg, S_lg, n_atoms, length(edges))
            g_est = (gnn_loss(pp_out,L_G) - gnn_loss(pm_out,L_G)) ./ (2 .* delta)
            new_t = deepcopy(gnn_params.theta); new_t[1] -= ak .* g_est
            gnn_params = LGNNParams(new_t, gnn_params.phi, gnn_params.J,
                                    gnn_params.feat_dim, gnn_params.T_iter,
                                    gnn_params.W_out, gnn_params.b_out)
            epoch % 50 == 0 && @printf("  epoch %3d  ar=%.4f\n", epoch, ar_e)
        end
        @printf("  LGNN standalone ar = %.4f\n", gnn_ar)
        mkpath(directoryBasis)
        serialize(gnn_file, (gnn_params, best_p_gnn, gnn_ar, n_atoms, n_edges, edges))
        println("  Saved LGNN → $(gnn_file)")
    end
    # [FIX] Delta_WS 不再需要 —— loss_fn_warmstart 现在内部自己按论文公式
    # 精确算Δ, 用 p_gnn 做的是"有偏初态制备", 不是修改失谐场。

    # ── Step③ (核信息热启动): 默认关闭! 只有在 `gdqc_validate` 模式的验证
    # 结果支持 (Pearson r(κ,gain) > 0.4 且 mean(gain) > 0) 之后, 才把下面
    # 这个开关改成 true。SIMILARITY_THRESHOLD 建议参考 gdqc_validate 打印
    # 出来的 "similarity_threshold ≈ ..." 建议值, 不要直接改成 0。
    ENABLE_KERNEL_WARMSTART = false
    SIMILARITY_THRESHOLD    = 0.6

    kernel_lib = GDQCLibEntry[]
    chi_new    = Float64[]
    if ENABLE_KERNEL_WARMSTART
        lib_path = joinpath(directoryBasis, "gdqc_library_p$(p_max).jls")
        # 注意: 库是按 p_val 存的 (gdqc_validate p_val [n_c_bins] 生成),
        # 如果你验证时用的 p_val 跟这里的 p_max 不一致, 请先针对每个要用的
        # p_val 都跑一遍 gdqc_validate, 或者手动改这里加载对应 p 的库文件。
        if isfile(lib_path)
            kernel_lib = deserialize(lib_path)
            chi_new = gdqc_signature(best_p_gnn, atoms, edges, n_atoms)
            println("  已加载核库: $(length(kernel_lib)) 条记录 ($(lib_path))")
        else
            println("  ⚠ 未找到核库 $(lib_path) —— kernel_warmstart 将退化为冷启动。")
            println("    请先运行: julia r3g_sge.jl gdqc_validate $(p_max)")
        end
    end

    # ── Step 2: p=1→p_max chain, 3(或4) conditions ──────────────────────────
    CONDITIONS = ENABLE_KERNEL_WARMSTART ?
        ["cold_start", "tqa_init", "gnn_warmstart", "kernel_warmstart"] :
        ["cold_start", "tqa_init", "gnn_warmstart"]
    chain_ar   = Dict(c => Dict{Int,Float64}() for c in CONDITIONS)
    chain_mean = Dict(c => Dict{Int,Float64}() for c in CONDITIONS)
    chain_std  = Dict(c => Dict{Int,Float64}() for c in CONDITIONS)
    prev_best  = Dict{String,Any}(c => nothing for c in CONDITIONS)   # [FIX] 显式Any类型, 否则Julia会把值类型锁死成Nothing

    for p_val in 1:p_max
        dim    = 2 * p_val
        lo     = zeros(dim)
        hi     = vcat([[3π, π] for _ in 1:p_val]...)
        budget = max(200, 100 * dim)
        n_init = max(20,    8 * dim)
        delta_t = 0.75
        tqa_angles = vcat([[delta_t*(k/p_val)*π, delta_t*(1-k/p_val)*π/2]
                           for k in 1:p_val]...)

        # Step③: 每个 p_val 只算一次 (跟 tqa_angles 一样, 不用每个 seed 重算)。
        # 相似度低于阈值时 kw_angles 保持 nothing, 该条件退化为纯冷启动 ——
        # 这正是"保守阈值"的意思: 宁可退化成冷启动, 也不要用不靠谱的种子点。
        kw_angles = nothing
        if ENABLE_KERNEL_WARMSTART && !isempty(kernel_lib)
            ws = kernel_warmstart(chi_new, kernel_lib, p_val; k=3)
            if ws !== nothing && ws.similarity > SIMILARITY_THRESHOLD
                kw_angles = ws.angles0
                @printf("  [kernel_warmstart] p=%d  similarity=%.3f  → 使用核库种子点\n",
                        p_val, ws.similarity)
            else
                sim_str = ws === nothing ? "无匹配维度" : @sprintf("%.3f", ws.similarity)
                println("  [kernel_warmstart] p=$(p_val)  similarity=$(sim_str) < 阈值$(SIMILARITY_THRESHOLD)  → 退化为冷启动")
            end
        end

        println("\n" * "─"^68)
        println("p=$(p_val)  dim=$(dim)D  budget=$(budget)  n_init=$(n_init)")
        println("─"^68)

        for cond in CONDITIONS
            println("\n  ── $(cond) ──")
            flush(stdout)

            task_params = [(task_id    = tid,
                            p_val      = p_val,
                            d_fixed    = d_fixed,
                            budget     = budget,
                            n_init     = n_init,
                            cond       = cond,
                            angle_hint = cond == "tqa_init" ? tqa_angles :
                                         (cond == "kernel_warmstart" ? kw_angles : nothing),
                            prev_hint  = prev_best[cond],
                            t_start    = time())
                           for tid in 1:length(SEEDS)]

            cond_results = pmap(task_params) do par
                seed = SEEDS_W[((par.task_id-1) % length(SEEDS_W)) + 1]
                Random.seed!(seed)
                cnd = par.cond

                cache = Dict{Vector{Float64},Float64}()
                function ev!(x)
                    xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
                    haskey(cache, key) && return cache[key]
                    try
                        lv = if cnd == "gnn_warmstart"
                            loss_fn_warmstart(xs, best_p_gnn, atoms, edges, n_atoms;
                                              graph_type=graph_type)
                        else
                            loss_fn(xs, atoms, edges, n_atoms)
                        end
                        v = isfinite(lv) ? clamp(lv, -(n_edges+1.0), 1000.0) : 1000.0
                        cache[key] = v; return v
                    catch; return 1000.0; end
                end

                halton_pts = halton_samples(par.n_init, lo, hi; skip=(seed*7)%100)
                init_xs = Vector{Vector{Float64}}(undef, par.n_init)
                if cnd == "tqa_init"
                    n_tqa = max(3, par.n_init÷3)
                    for i in 1:n_tqa
                        sc = 0.05 + 0.10*(i-1)/n_tqa
                        init_xs[i] = clamp.(par.angle_hint .+ randn(dim).*sc.*(hi.-lo), lo, hi)
                    end
                    for i in (n_tqa+1):par.n_init
                        init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                    end
                elseif cnd == "kernel_warmstart" && !isnothing(par.angle_hint) &&
                       length(par.angle_hint) == dim
                    n_kw = max(3, par.n_init÷3)
                    for i in 1:n_kw
                        sc = 0.05 + 0.10*(i-1)/n_kw
                        init_xs[i] = clamp.(par.angle_hint.+randn(dim).*sc.*(hi.-lo), lo, hi)
                    end
                    for i in (n_kw+1):par.n_init
                        init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                    end
                elseif !isnothing(par.prev_hint) && length(par.prev_hint)==dim
                    n_h = max(3, par.n_init÷3)
                    for i in 1:n_h
                        sc = 0.05 + 0.10*(i-1)/n_h
                        init_xs[i] = clamp.(par.prev_hint.+randn(dim).*sc.*(hi.-lo), lo, hi)
                    end
                    for i in (n_h+1):par.n_init
                        init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                    end
                else
                    for i in 1:par.n_init
                        init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                    end
                end
                init_ys = Float64[ev!(x) for x in init_xs]
                X=hcat(init_xs...); y=copy(init_ys)
                by=minimum(init_ys); bx=init_xs[argmin(init_ys)]

                Q_g=2; Q_c=2
                n_cand       = dim>=8 ? 800 : 400
                map_every    = dim>=8 ? 12 : 18
                phase1_end   = dim>=8 ? 0.45 : 0.30
                phase2_end   = dim>=8 ? 0.78 : 0.68
                nr_hp        = dim>=8 ? 4 : 2
                stag_limit   = dim>=8 ? 25 : 20
                refine_every = 15
                nm_iters     = dim>=8 ? 200 : 100
                refine_scales = dim>=8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
                θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp); stag=0

                for iter in 1:(par.budget-par.n_init)
                    t=iter/(par.budget-par.n_init)
                    iter>1 && iter%map_every==0 &&
                        (θ_gp=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp))
                    gp=build_spectral_gp(X,y,θ_gp,Q_g,Q_c,dim)
                    C=lo.+rand(dim,n_cand).*(hi.-lo)
                    if t<phase1_end
                        xn=C[:,ts_acq(gp,C;ns=10)]
                    elseif t<phase2_end
                        ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
                        μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
                    else
                        β=max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
                        μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]
                    end
                    yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
                    if yn<by; by=yn; bx=copy(xn); stag=0
                    else; t>=phase1_end && (stag+=1); end
                    if stag>=stag_limit
                        for _ in 1:2
                            xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                            X=hcat(X,xr); push!(y,yr); yr<by && (by=yr;bx=copy(xr))
                        end; stag=0
                    end
                    if iter%refine_every==0 && t>=phase1_end
                        for sc in refine_scales
                            xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                            r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                                       Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                            yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                            yr<by-1e-6 && (by=yr;bx=xr;stag=0;X=hcat(X,xr);push!(y,yr))
                        end
                    end
                end
                ar=-by/n_edges
                println("    task=$(par.task_id) [$(cnd)] p=$(p_val) ar=$(round(ar,digits=4))")
                (par.task_id, p_val, d_fixed, seed, cnd, by, bx, ar)
            end

            ars = [r[8] for r in cond_results]
            chain_ar[cond][p_val]   = maximum(ars)
            chain_mean[cond][p_val] = mean(ars)
            chain_std[cond][p_val]  = std(ars)
            @printf("  [%s] p=%d: mean=%.4f ± %.4f  best=%.4f\n",
                    cond, p_val, chain_mean[cond][p_val],
                    chain_std[cond][p_val], chain_ar[cond][p_val])

            best_r = argmax(r->r[8], cond_results)
            prev_best[cond] = best_r[7]

            println("  ┌─ p=$(p_val) snapshot ─────────────────────────────────────────────┐")
            for c2 in CONDITIONS
                haskey(chain_ar[c2], p_val) || continue
                @printf("  │  %-16s  best=%.4f  mean=%.4f ± %.4f  │\n",
                        c2, chain_ar[c2][p_val], chain_mean[c2][p_val], chain_std[c2][p_val])
            end
            println("  └──────────────────────────────────────────────────────────────────┘")
        end
    end

    mkpath(directoryBasis)
    save_tag = "init_compare_$(graph_type)_n$(n_atoms)_d$(d_fixed)"
    serialize(joinpath(directoryBasis, string(save_tag, ".jls")),
              (chain_ar, chain_mean, chain_std, CONDITIONS, label, n_edges))
    println("\n✓ Saved → " * joinpath(directoryBasis, save_tag * ".jls"))

    println("\n\n" * "═"^70)
    println("INITIALISATION COMPARISON — FINAL SUMMARY")
    println("$(label)  d=$(d_fixed) μm  $(length(SEEDS)) seeds  GW=0.878")
    println("═"^70)
    header_cols = ENABLE_KERNEL_WARMSTART ?
        "  p │ dim │   cold_start        │   tqa_init          │   gnn_warmstart     │   kernel_warmstart" :
        "  p │ dim │   cold_start        │   tqa_init          │   gnn_warmstart"
    println(header_cols)
    println("    │     │  best   mean±std    │  best   mean±std    │  best   mean±std    " *
            (ENABLE_KERNEL_WARMSTART ? "│  best   mean±std" : ""))
    println("  " * "─"^68)
    for p_val in 1:p_max
        base_str = @sprintf("  %d │  %2dD │ %.4f  %.4f±%.4f │ %.4f  %.4f±%.4f │ %.4f  %.4f±%.4f",
            p_val, 2*p_val,
            chain_ar["cold_start"][p_val],   chain_mean["cold_start"][p_val],   chain_std["cold_start"][p_val],
            chain_ar["tqa_init"][p_val],     chain_mean["tqa_init"][p_val],     chain_std["tqa_init"][p_val],
            chain_ar["gnn_warmstart"][p_val],chain_mean["gnn_warmstart"][p_val],chain_std["gnn_warmstart"][p_val])
        if ENABLE_KERNEL_WARMSTART && haskey(chain_ar["kernel_warmstart"], p_val)
            base_str *= @sprintf(" │ %.4f  %.4f±%.4f",
                chain_ar["kernel_warmstart"][p_val], chain_mean["kernel_warmstart"][p_val],
                chain_std["kernel_warmstart"][p_val])
        end
        println(base_str)
    end
    println("  " * "─"^68)
    println("  LGNN standalone ar = $(round(gnn_ar, digits=4))")
    println("═"^70)
    exit(0)
end


# ── LARGER GRAPH BO MODE ──────────────────────────────────────────────────────
# julia Q3_sge.jl larger_graph <graph_type> <n_atoms> <p_val> [d_fixed]
# Runs the full GSM+CSM BO (same as pscale but on a larger graph)
# Graph types: q4 (16-atom), reg3 (3-regular random), udisk (unit-disk)
# ── LARGER_GRAPH_CHAIN MODE ──────────────────────────────────────────────────
# julia Q3_sge.jl larger_graph_chain <graph_type> <n_atoms> [d_init] [d_step] [n_grid]
#
# Adaptive p=1→5 chain on ANY supported graph type, with d-grid search at each p.
# Identical logic to pscale_chain but:
#   • Accepts graph_type (q4 | reg3 | udisk) and n_atoms
#   • Rebuilds the target graph on each worker (graph_type + n_atoms passed in params)
#   • Reports ar relative to n_edges (not hardcoded 12)
#   • Saves per-p results tagged with graph_type and n_atoms
#
# Example:
#   julia Q3_sge.jl larger_graph_chain q4  16 14.0 0.5 5   # Q4 hypercube, p=1→5
#   julia Q3_sge.jl larger_graph_chain reg3 16 14.0 0.5 5  # 3-regular n=16
#   julia Q3_sge.jl larger_graph_chain reg3 20 14.0 0.5 5  # 3-regular n=20
#   julia Q3_sge.jl larger_graph_chain udisk 16 14.0 0.5 5 # unit-disk n=16

if length(ARGS) >= 1 && ARGS[1] == "larger_graph_chain"
    graph_type = length(ARGS) >= 2 ? ARGS[2] : "q4"
    n_at       = length(ARGS) >= 3 ? parse(Int,     ARGS[3]) : 16
    d_centre_0 = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 14.0
    d_step     = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.5
    n_grid     = length(ARGS) >= 6 ? parse(Int,     ARGS[6]) : 5
    using Statistics

    # ── Verify graph can be built before spawning any workers ──────────────
    test_atoms, test_edges, n_atoms, n_edges, graph_label =
        make_graph(graph_type, n_at, d_centre_0; seed=42)

    println("\n" * "="^72)
    println("ADAPTIVE P-SCALING CHAIN — LARGER GRAPH")
    println("Graph:    $(graph_label)  |V|=$(n_atoms)  |E|=$(n_edges)")
    println("d-centre: $(d_centre_0) μm  step: $(d_step) μm  grid: $(n_grid) pts")
    println("Seeds:    $(length(SEEDS))  per d-value")
    println("GW bound: $(round(0.878*n_edges, digits=2)) expected cut")
    println("="^72)

    # Budget table
    println("\nBudget schedule:")
    for p in 1:5
        dim = 2*p
        bud = max(QAOA_BUDGET, 100*dim); ni = max(QAOA_N_INIT, 8*dim)
        nc  = dim >= 8 ? 800 : 400
        @printf("  p=%d  dim=%2dD  budget=%4d  n_init=%3d  n_cand=%d\n",
                p, dim, bud, ni, nc)
    end
    println()

    chain_results = Dict{Int, Any}()
    mkpath(directoryBasis)

    let d_centre = d_centre_0, d_step = d_step, n_grid = n_grid
    for p_val in 1:5
        # ── Build d-grid centred at current best d ─────────────────────────
        half = (n_grid - 1) ÷ 2
        d_grid = [d_centre + i*d_step for i in -half:half]
        d_grid = sort(unique(clamp.(round.(d_grid, digits=3), 8.0, 22.0)))

        dim_p    = 2 * p_val
        ps_bud   = max(QAOA_BUDGET, 100 * dim_p)
        ps_init  = max(QAOA_N_INIT,   8 * dim_p)

        # ── Warm-start hint from p-1 best params ──────────────────────────
        x0_hint_chain = nothing
        if p_val > 1 && haskey(chain_results, p_val-1)
            prev_bx  = chain_results[p_val-1].best_params
            prev_dim = length(prev_bx)
            if prev_dim == 2*(p_val-1)
                γ_mean = mean(prev_bx[1:2:prev_dim])
                β_mean = mean(prev_bx[2:2:prev_dim])
                x0_hint_chain = vcat(prev_bx, [γ_mean, β_mean])
                println("  x0_hint: extended p=$(p_val-1)→$(p_val)  dim=$(prev_dim)→$(dim_p)")
            end
        end

        n_tasks = length(d_grid) * length(SEEDS)
        println("\n── p=$(p_val)  d-grid=$(d_grid)  budget=$(ps_bud)  dim=$(dim_p)D  tasks=$(n_tasks) ──")
        flush(stdout)

        # ── Dispatch: one task per (d_value × seed) ────────────────────────
        all_params = [(task_id    = (di-1)*length(SEEDS) + si,
                       p_val      = p_val,
                       d_fixed    = d_grid[di],
                       budget     = ps_bud,
                       n_init     = ps_init,
                       x0_hint    = x0_hint_chain,
                       t_start    = time(),
                       graph_type = graph_type,
                       n_at       = n_at)
                      for di in 1:length(d_grid)
                      for si in 1:length(SEEDS)]

        # ── Worker: compute_task_p but with per-graph atoms/edges ──────────
        @time p_results = pmap(all_params) do par
            seed = SEEDS_W[((par.task_id-1) % length(SEEDS_W)) + 1]
            Random.seed!(seed)

            # Build graph on this worker
            at, ed, na, ne, _ = make_graph(par.graph_type, par.n_at, par.d_fixed; seed=42)

            p_val = par.p_val; d = par.d_fixed
            budget = par.budget; n_init = par.n_init
            x0_hint = par.x0_hint
            dim  = 2 * p_val
            lo   = zeros(dim); hi = vcat([[3π, π] for _ in 1:p_val]...)

            # ev! caches evaluations, clamps output to valid range
            cache = Dict{Vector{Float64},Float64}()
            eval_count = Ref(0)
            function ev!(x)
                xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
                haskey(cache, key) && return cache[key]
                eval_count[] += 1
                try
                    lv = loss_fn(xs, at, ed, na)
                    v  = isfinite(lv) ? clamp(lv, -(ne+1.0), 1000.0) : 1000.0
                    cache[key] = v; return v
                catch; return 1000.0; end
            end

            # Halton warm-start with optional x0_hint (from p-1 best)
            n_h = n_init
            halton_pts = halton_samples(n_h, lo, hi; skip=(seed*7)%100)
            init_xs = Vector{Vector{Float64}}(undef, n_h)
            if !isnothing(x0_hint) && length(x0_hint) == dim
                n_hint = max(3, n_h÷3)
                for i in 1:n_hint
                    sc = 0.05 + 0.10*(i-1)/n_hint
                    init_xs[i] = clamp.(x0_hint .+ randn(dim).*sc.*(hi.-lo), lo, hi)
                end
                for i in (n_hint+1):n_h
                    init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                end
            else
                for i in 1:n_h
                    init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                end
            end
            init_ys = Float64[ev!(x) for x in init_xs]
            X = hcat(init_xs...); y = copy(init_ys)
            by = minimum(init_ys); bx = init_xs[argmin(init_ys)]
            println("  [$(par.graph_type) p=$(p_val) seed=$(seed) d=$(d)μm] warm-start ar=$(round(-by/ne,digits=4))")
            flush(stdout)

            # BO hyperparameters (same as compute_task_p proven settings)
            Q_g=2; Q_c=2
            n_cand       = dim >= 8 ? 800  : 400
            map_every    = dim >= 8 ? 12   : 18
            phase1_end   = dim >= 8 ? 0.45 : 0.30
            phase2_end   = dim >= 8 ? 0.78 : 0.68
            nr_hp        = dim >= 8 ? 4    : 2
            stag_limit   = dim >= 8 ? 25   : 20
            refine_every = 15
            nm_iters     = dim >= 8 ? 200  : 100
            refine_scales = dim >= 8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
            θ_gp = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=nr_hp); stag=0
            n_bo = budget - n_init

            for iter in 1:n_bo
                t = iter/n_bo; eval_idx = min(n_init+iter, budget)
                iter>1 && iter%map_every==0 &&
                    (θ_gp = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=nr_hp))
                gp = build_spectral_gp(X, y, θ_gp, Q_g, Q_c, dim)
                C  = lo .+ rand(dim, n_cand).*(hi.-lo)
                if t < phase1_end
                    xn=C[:,ts_acq(gp,C;ns=10)]; lab="TS"
                elseif t < phase2_end
                    ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
                    μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]; lab="EI"
                else
                    β=max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
                    μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]; lab="LCB"
                end
                yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
                if yn<by
                    by=yn; bx=copy(xn); stag=0
                    println("    eval=$(n_init+iter) [$lab] ✓ ar=$(round(-by/ne,digits=4))")
                    flush(stdout)
                else; t>=phase1_end && (stag+=1); end
                if stag>=stag_limit
                    for _ in 1:2
                        xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                        X=hcat(X,xr); push!(y,yr); yr<by && (by=yr;bx=copy(xr))
                    end; stag=0
                end
                if iter%refine_every==0 && t>=phase1_end
                    for sc in refine_scales
                        xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                        r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                                   Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                        yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                        if yr<by-1e-6
                            by=yr; bx=xr; stag=0; X=hcat(X,xr); push!(y,yr)
                            println("    [MRLR sc=$(sc)] ✓ ar=$(round(-by/ne,digits=4))")
                        end
                    end
                end
            end
            ar=-by/ne; elapsed=round(time()-par.t_start,digits=1)
            println("  done  p=$(p_val)  seed=$(seed)  d=$(d)μm  ar=$(round(ar,digits=4))  $(elapsed)s")
            flush(stdout)
            (par.task_id, p_val, d, seed, by, bx, eval_count[], ne, ar, elapsed)
        end   # pmap

        # ── Per-d statistics ───────────────────────────────────────────────
        # ── d* = d with highest MEAN ar across seeds (robust to outliers) ──
        d_mean_ar = Dict{Float64,Float64}()
        for d_val in d_grid
            d_ars = [r[9] for r in p_results if abs(r[3]-d_val) < 0.01]
            isempty(d_ars) || (d_mean_ar[d_val] = mean(d_ars))
        end
        best_d   = argmax(d_mean_ar)          # d with highest mean ar
        # best_ar and best_bx = from the single best seed at best_d
        best_r   = argmax(r->r[9], [r for r in p_results if abs(r[3]-best_d) < 0.01])
        best_ar  = best_r[9]; best_bx = best_r[6]

        println("\n  Results p=$(p_val)  graph=$(graph_label):")
        println("  " * "─"^62)
        for d_val in d_grid
            d_ars = [r[9] for r in p_results if abs(r[3]-d_val) < 0.01]
            isempty(d_ars) && continue
            mark = abs(d_val - best_d) < 0.01 ? "  ← d*" : ""
            @printf("  d=%5.2f μm: mean=%.4f  std=%.4f  best=%.4f  P>0.80=%.2f  (n=%d)%s\n",
                d_val, mean(d_ars), std(d_ars), maximum(d_ars),
                mean(d_ars .> 0.80), length(d_ars), mark)
        end
        println("  " * "─"^62)
        gw_cut = 0.878 * n_edges
        @printf("  p=%d best: d*=%.3f μm  ar=%.4f  cut=%.2f/%.0f  gap=%+.1f%%\n",
            p_val, best_d, best_ar, best_ar*n_edges, Float64(n_edges),
            (best_ar-0.878)/0.878*100)
        @printf("       γ,β = %s\n", string(round.(best_bx, digits=4)))
        flush(stdout)

        # Save
        tag = "lg_chain_$(graph_type)_n$(n_at)_p$(p_val)_d$(round(best_d,digits=2))"
        serialize(joinpath(directoryBasis,"$(tag).jls"), p_results)
        println("  ✓ Saved → " * joinpath(directoryBasis, tag * ".jls"))

        chain_results[p_val] = (best_ar=best_ar, best_d=best_d,
                                 best_params=best_bx, all_results=p_results)
        # Update d_centre for next p
        d_centre = best_d
        p_val < 5 && println("  → next p=$(p_val+1) grid centred at d=$(round(d_centre,digits=3)) μm")
    end   # for p_val
    end   # let d_centre

    # ── Final summary ─────────────────────────────────────────────────────────
    println("\n\n" * "═"^72)
    println("ADAPTIVE P-SCALING CHAIN — FINAL SUMMARY")
    println("Graph: $(graph_label)  |V|=$(n_atoms)  |E|=$(n_edges)  GW cut ≥ $(round(0.878*n_edges,digits=1))")
    println("═"^72)
    println("  p │ dim │  d* (μm) │  best ar │  cut/$(n_edges) │  gap to GW  │  γ₁, β₁ (rad)")
    println("  " * "─"^70)
    GW = 0.878
    for p_val in 1:5
        haskey(chain_results, p_val) || continue
        r = chain_results[p_val]
        gap = (r.best_ar - GW)/GW * 100
        px  = r.best_params
        @printf("  %d │  %2dD │  %6.3f  │  %.4f  │  %5.2f    │  %+6.1f%%     │  %.3f, %.3f\n",
            p_val, 2*p_val, r.best_d, r.best_ar, r.best_ar*n_edges,
            gap, px[1], px[2])
    end
    println("  " * "═"^70)
    exit(0)
end

if length(ARGS) >= 1 && ARGS[1] == "larger_graph"
    graph_type = length(ARGS) >= 2 ? ARGS[2] : "q4"
    n_at       = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
    p_val      = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 3
    d_fixed    = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 14.0
    n_seeds    = length(SEEDS)
    using Statistics

    println("\n" * "="^65)
    println("LARGER GRAPH BO: type=$(graph_type)  n=$(n_at)  p=$(p_val)  d=$(d_fixed)μm")
    println("Seeds: $(n_seeds)  Budget: $(max(200, 100*2*p_val))  Dim: $(2*p_val)D")
    println("="^65)

    # Build graph (use seed 42 for fixed random graph)
    atoms, edges, n_atoms, n_edges, label = make_graph(graph_type, n_at, d_fixed; seed=42)
    println("Graph: $(label)  |V|=$(n_atoms)  |E|=$(n_edges)")
    opt_cut_bound = n_edges  # upper bound (not necessarily tight for non-bipartite)
    println("Max possible cut: $(opt_cut_bound) (edges); GW bound: $(round(0.878*n_edges, digits=1))")

    # Dispatch tasks
    ps_budget = max(200, 100 * 2 * p_val)
    ps_n_init = max(20,    8 * 2 * p_val)
    params_lg = [(task_id = tid,
                  p_val   = p_val,
                  d_fixed = d_fixed,
                  budget  = ps_budget,
                  n_init  = ps_n_init,
                  x0_hint = nothing,
                  t_start = time(),
                  # Pass graph info via closure — atoms/edges built per worker
                  graph_type = graph_type,
                  n_at       = n_at)
                 for tid in 1:n_seeds]

    # Worker function for larger graphs (defined @everywhere earlier as compute_task_p)
    # We override the atoms/edges inside each worker call
    lg_results = pmap(params_lg) do par
        Random.seed!(SEEDS_W[((par.task_id-1) % length(SEEDS_W)) + 1])
        at, ed, na, ne, _ = make_graph(par.graph_type, par.n_at, par.d_fixed; seed=42)
        # Build a modified params with n_edges patched
        mod_par = (task_id=par.task_id, p_val=par.p_val, d_fixed=par.d_fixed,
                   budget=par.budget, n_init=par.n_init,
                   x0_hint=par.x0_hint, t_start=par.t_start)
        # Use compute_task_p but with overridden graph
        dim = 2 * par.p_val
        seed = SEEDS_W[((par.task_id-1) % length(SEEDS_W)) + 1]
        lo  = zeros(dim); hi = vcat([[3π, π] for _ in 1:par.p_val]...)
        budget = par.budget; n_init = par.n_init

        cache = Dict{Vector{Float64},Float64}(); eval_count = Ref(0)
        function ev!(x)
            xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
            haskey(cache, key) && return cache[key]
            eval_count[] += 1
            try
                lv = loss_fn(xs, at, ed, na)
                v  = isfinite(lv) ? clamp(lv, -(ne+1.0), 1000.0) : 1000.0
                cache[key] = v; return v
            catch; return 1000.0; end
        end

        n_h = n_init
        halton_pts = halton_samples(n_h, lo, hi; skip=seed%50)
        init_xs = [clamp.(pt .+ randn(dim).*0.05.*(hi.-lo), lo, hi) for pt in halton_pts]
        init_ys = Float64[ev!(x) for x in init_xs]
        X = hcat(init_xs...); y = copy(init_ys)
        by = minimum(init_ys); bx = init_xs[argmin(init_ys)]

        n_bo = budget - n_init
        Q_g=2; Q_c=2
        n_cand     = dim >= 8 ? 800 : 400
        map_every  = dim >= 8 ? 12 : 18
        phase1_end = dim >= 8 ? 0.45 : 0.30
        phase2_end = dim >= 8 ? 0.78 : 0.68
        nr_hp      = dim >= 8 ? 4 : 2
        stag_limit = dim >= 8 ? 25 : 20
        refine_every = 15; nm_iters = dim >= 8 ? 200 : 100
        refine_scales = dim >= 8 ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
        θ_gp = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=nr_hp); stag = 0

        for iter in 1:n_bo
            t = iter/n_bo; eval_idx = min(n_init+iter, budget)
            iter>1 && iter%map_every==0 &&
                (θ_gp = map_spectral_hp(X, y, Q_g, Q_c, dim; nr=nr_hp))
            gp = build_spectral_gp(X, y, θ_gp, Q_g, Q_c, dim)
            C  = lo .+ rand(dim, n_cand).*(hi.-lo)
            if t < phase1_end
                xn = C[:,ts_acq(gp,C;ns=10)]; lab="TS"
            elseif t < phase2_end
                ξ = 0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
                μ,σ = gp_pred(gp,C); xn = C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]; lab="EI"
            else
                β = max(exp(-2.5*(t-phase2_end)/(1-phase2_end)), 0.05)
                μ,σ = gp_pred(gp,C); xn = C[:,argmin(lcb_acq(μ,σ;β=β))]; lab="LCB"
            end
            yn = ev!(xn); X=hcat(X,xn); push!(y,yn)
            if yn < by; by=yn; bx=copy(xn); stag=0
            else; t>=phase1_end && (stag+=1); end
            if stag >= stag_limit
                for _ in 1:2
                    xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                    X=hcat(X,xr); push!(y,yr); yr<by && (by=yr; bx=copy(xr))
                end; stag=0
            end
            if iter%refine_every==0 && t>=phase1_end
                for sc in refine_scales
                    xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                    r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                               Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                    yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                    yr<by-1e-6 && (by=yr; bx=xr; stag=0; X=hcat(X,xr); push!(y,yr))
                end
            end
        end
        ar=-by/ne; elapsed=round(time()-par.t_start,digits=1)
        println("  done  task=$(par.task_id)  ar=$(round(ar,digits=4))  $(elapsed)s")
        (par.task_id, par.p_val, d_fixed, seed, by, bx, eval_count[], ne, ar, elapsed)
    end

    ars = [r[9] for r in lg_results]
    println("\n" * "═"^65)
    println("LARGER GRAPH RESULTS: $(label)  p=$(p_val)  d=$(d_fixed)μm")
    println("═"^65)
    @printf("  mean ar = %.4f ± %.4f\n", mean(ars), std(ars))
    @printf("  best ar = %.4f  (GW bound = %.4f)\n", maximum(ars), 0.878)
    @printf("  gap to GW = %+.4f (%+.1f%%)\n",
            maximum(ars)-0.878, (maximum(ars)-0.878)/0.878*100)
    @printf("  P(ar > 0.80) = %.2f\n", mean(ars .> 0.80))

    best_r = argmax(r->r[9], lg_results)
    println("  Best params: $(round.(best_r[6], digits=4))")

    mkpath(directoryBasis)
    tag = "larger_$(graph_type)_n$(n_at)_p$(p_val)"
    serialize(joinpath(directoryBasis, "$(tag).jls"), lg_results)
    println("✓ Saved → " * joinpath(directoryBasis, tag * ".jls"))
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── GDQC 验证模式 (Step ①+②): 核相似度是否预测 QAOA 角度可迁移性 ──────────
# 运行方式:
#   julia r3g_sge.jl gdqc_validate [p_val] [n_c_bins]
#   例如: julia r3g_sge.jl gdqc_validate 3 10
#
# 不需要 addprocs/Distributed —— 图很小, 单进程跑完全套图 + 有序对通常是
# 几分钟量级 (实际耗时取决于 Bloqade 模拟速度, 可先减少 GRAPH_SPECS 试跑)。
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "gdqc_validate"
    using Statistics, LinearAlgebra, Printf

    p_val    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    n_c_bins = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10

    println("\n" * "="^70)
    println("GDQC 验证实验: 核相似度是否预测 QAOA 角度可迁移性")
    println("p_val = $(p_val)  (dim = $(2*p_val)D)   n_c_bins = $(n_c_bins)")
    println("="^70)

    # 待比较的图集合: 覆盖不同拓扑 + 不同大小。可按需增删, 建议至少 5~6 张,
    # 才能算出有统计意义的相关系数 (会生成 n*(n-1) 个有序对)。
    GRAPH_SPECS = [
        ("q3",    8, 14.0),
        ("q4",   16, 14.0),
        ("reg3",  8, 14.0),
        ("reg3", 10, 14.0),
        ("udisk", 10, 14.0),
        ("udisk", 12, 14.0),
        ("honeycomb", 10, 14.0),
        ("honeycomb", 14, 14.0),
    ]

    println("\n[Step ①+②] 为每张图: 训练 LGNN → 提取 GDQC 特征 χ(G) → 求自身最优角度\n")

    entries = GDQCLibEntry[]
    for (gtype, n_at, d) in GRAPH_SPECS
        atoms, edges, n_atoms, n_edges, label = make_graph(gtype, n_at, d; seed=42)
        print("  $(label)  |V|=$(n_atoms) |E|=$(n_edges)  ... ")
        flush(stdout)

        gnn_file = joinpath(directoryBasis, "gnn_$(gtype)_n$(n_atoms).jls")
        local p_gnn
        if isfile(gnn_file)
            _, p_gnn, _, _, _, _ = deserialize(gnn_file)
        else
            gnn_params_, p_gnn, gnn_ar = train_lgnn_quick(edges, n_atoms; n_epochs=300)
            mkpath(directoryBasis)
            serialize(gnn_file, (gnn_params_, p_gnn, gnn_ar, n_atoms, n_edges, edges))
        end

        chi = gdqc_signature(p_gnn, atoms, edges, n_atoms; n_c_bins=n_c_bins)
        best_bx, best_ar = quick_best_angles_bo(atoms, edges, n_atoms, p_val)
        cold_ar = random_angle_baseline(atoms, edges, n_atoms, p_val; n_samples=20)
        _, gnn_solve_ar = quick_best_angles_bo_gnn(atoms, edges, n_atoms, p_gnn, p_val, gtype)

        @printf("own_ar=%.4f  cold_ar=%.4f  gnn_ar=%.4f  |χ|=%d\n",
                best_ar, cold_ar, gnn_solve_ar, length(chi))
        push!(entries, GDQCLibEntry(gtype, n_atoms, d, label, n_edges, chi,
                                     best_bx, best_ar, cold_ar, gnn_solve_ar, atoms, edges))
    end

    mkpath(directoryBasis)
    lib_path = joinpath(directoryBasis, "gdqc_library_p$(p_val).jls")
    serialize(lib_path, entries)
    println("\n✓ 核库已保存 → $(lib_path)")

    println("\n[Step ②] 零样本角度迁移实验: 遍历所有有序对 (A→B)\n")

    rows = NamedTuple[]
    for A in entries, B in entries
        A === B && continue
        kappa = kappa_gdqc(A.chi, B.chi)
        lv = try
            loss_fn(A.best_bx, B.atoms, B.edges, B.n_atoms)
        catch
            1000.0
        end
        ar_transfer = isfinite(lv) ? -clamp(lv, -(B.n_edges+1.0), 1000.0)/B.n_edges : 0.0
        gain = ar_transfer - B.cold_ar
        push!(rows, (A=A.label, B=B.label, kappa=kappa, ar_transfer=ar_transfer,
                      cold_ar_B=B.cold_ar, gain=gain, own_ar_A=A.best_ar, own_ar_B=B.best_ar,
                      gnn_ar_A=A.gnn_ar, gnn_ar_B=B.gnn_ar))
    end

    csv_path = joinpath(directoryBasis, "gdqc_validation_pairs_p$(p_val).csv")
    open(csv_path, "w") do io
        println(io, "graph_A,graph_B,kappa_gdqc,ar_transfer,cold_ar_B,transfer_gain,own_ar_A,own_ar_B,gnn_ar_A,gnn_ar_B")
        for r in rows
            @printf(io, "%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.A, r.B, r.kappa, r.ar_transfer, r.cold_ar_B, r.gain, r.own_ar_A, r.own_ar_B,
                    r.gnn_ar_A, r.gnn_ar_B)
        end
    end
    println("✓ 明细表已保存 → $(csv_path)")

    kappas = [r.kappa for r in rows]
    gains  = [r.gain  for r in rows]
    pearson_r = length(rows) >= 3 ? cor(kappas, gains) : NaN

    println("\n" * "="^70)
    println("验证结果汇总  (共 $(length(rows)) 个有序图对)")
    println("="^70)
    @printf("  κ_GDQC 范围     : [%.4f, %.4f]\n", minimum(kappas), maximum(kappas))
    @printf("  transfer_gain   : mean=%.4f  std=%.4f  (>0 表示比随机基线好)\n",
            mean(gains), std(gains))
    @printf("  Pearson r(κ, gain) = %.4f\n", pearson_r)
    println("─"^70)

    if pearson_r > 0.4 && mean(gains) > 0
        println("  判定: 验证通过 ✓ —— κ_GDQC 与迁移增益正相关且总体收益为正。")
        println("        建议进入 Step③: 把 kernel_warmstart 接入 gnn_qaoa 模式,")
        println("        相似度阈值建议先设 similarity_threshold ≈ ",
                round(quantile(kappas,0.7),digits=3), "\n")
    elseif pearson_r > 0.15
        println("  判定: 弱相关 —— 有一定信号但不够强。")
        println("        建议: 先扩大 GRAPH_SPECS 的图数量 (尤其增加同拓扑不同")
        println("        参数的图, 比如多个不同 seed 的 reg3), 再跑一次验证;")
        println("        如仍然只是弱相关, 接入③时把相似度阈值设得更保守")
        println("        (比如只用 top-1 最近邻而不是 top-3 加权平均)。\n")
    else
        println("  判定: 未通过验证 —— κ_GDQC 目前不能可靠预测角度可迁移性。")
        println("        不建议接入 Step③ 的 kernel_warmstart。")
        println("        但 χ(G) / κ_GDQC 仍可作为诊断工具使用 (分析 Q3→Q4")
        println("        近似比下降原因、给现有图实例做相似度聚类等)。\n")
    end
    println("="^70)
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── inspect_gnn_weighted: 直接读缓存, 打印这张具体图上GNN的p_i分布 ────────
# 不重新训练、不重新构图 —— 直接加载 gnn_qaoa_weighted 已经存好的缓存文件,
# 打印每个原子的p_i和对应划分。用来诊断"GNN warmstart表现不好, 到底是
# GNN本身没训好(p_i大量接近0.5), 还是别的原因"。
#
# 运行方式:
#   julia r3g_sge.jl inspect_gnn_weighted <n_atoms> [seed] [geom_mode] [k_mode]
# 例如:
#   julia r3g_sge.jl inspect_gnn_weighted 10 42 force3d
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "inspect_gnn_weighted"
    n_at      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    seed      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 42
    geom_mode = length(ARGS) >= 4 ? ARGS[4] : "force3d"
    k_mode    = length(ARGS) >= 5 ? ARGS[5] : "legacy"

    cache_tag = geom_mode == "force" ? "$(geom_mode)_$(k_mode)" : geom_mode
    gnn_file = joinpath(directoryBasis, "gnn_reg3w_n$(n_at)_seed$(seed)_$(cache_tag).jls")

    if !isfile(gnn_file)
        println("找不到缓存文件: $(gnn_file)")
        println("先跑一次 gnn_qaoa_weighted 让它训练并存缓存, 再回来看这个诊断。")
        exit(1)
    end

    gnn_params, best_p_gnn, gnn_ar, n_atoms, n_edges, edges = deserialize(gnn_file)

    println("\n" * "="^60)
    println("GNN 缓存诊断: $(gnn_file)")
    println("="^60)
    @printf("gnn_ar (存档时的离散cut) = %.4f\n", gnn_ar)
    println("\n每个原子的 p_i 分布:")
    local n_confident, n_uncertain
    n_confident = 0   # p接近0或1 (>0.85 或 <0.15)
    n_uncertain = 0   # p接近0.5 (0.35~0.65之间)
    for i in 1:n_atoms
        p = best_p_gnn[i]
        tag = p > 0.85 || p < 0.15 ? "确信" : (p > 0.35 && p < 0.65 ? "模糊" : "中等")
        tag == "确信" && (n_confident += 1)
        tag == "模糊" && (n_uncertain += 1)
        partition = p > 0.5 ? "A" : "B"
        @printf("  atom %2d: p=%.3f → %s  [%s]\n", i, p, partition, tag)
    end
    println("\n汇总: 确信(p>0.85或<0.15) = $(n_confident)/$(n_atoms)  ·  模糊(0.35<p<0.65) = $(n_uncertain)/$(n_atoms)")
    n_uncertain > n_atoms ÷ 3 &&
        println("⚠ 模糊原子占比较高 —— GNN对这张图的划分没有形成清晰共识, 有偏初态制备")
    println("  的偏置力度会被这些模糊原子明显稀释, warmstart机制的效果会打折扣。")
    println("="^60)
    exit(0)
end
# build_spectral_gp/ts_acq/ei_acq/lcb_acq/map_spectral_hp), 只做三处替换:
#   1. 图: make_reg3_weighted_graph + embed_weighted_graph_2d (带权重的3-regular图)
#   2. GNN 训练: train_lgnn_quick_weighted (用加权拉普拉斯矩阵)
#   3. ev!() 里调用 loss_fn_weighted / loss_fn_warmstart_weighted, ar 归一化
#      改用 sum(weights) 而不是 n_edges
# 只保留 cold_start / tqa_init / gnn_warmstart 三个条件 (kernel_warmstart 那个
# 实验性功能在加权版本里不重复实现)。
#
# 运行方式:
#   julia r3g_sge.jl gnn_qaoa_weighted <n_atoms> [p_max] [seed] [geom_mode] [d_fixed] [k_mode]
#
#   geom_mode = "force"(默认) → make_reg3_weighted_graph_force
#               连续力场嵌入, k_mode 两种(仅 force 模式下生效):
#                 k_mode="legacy"(默认) → 两个独立常数(k_rep=C6/4真实物理值,
#                                          k_attr由孤立边对平衡点反解), 实测
#                                          残差和权重比值三轮都最稳定
#                 k_mode="global"        → 论文Eq.6-7, 排斥/引力共用同一个k
#   geom_mode = "jitter"       → make_honeycomb_graph_jittered
#               拓扑固定用蜂窝格天然邻接关系, 位置=蜂窝格点(格点间距=d_fixed)
#               +小幅度高斯抖动(标准差=d_fixed*jitter_frac, 第7个参数位复用
#               成jitter_frac, 默认0.15)。没有优化步骤, 不存在embedding残差
#               这个概念, 从根上避开"独立随机拓扑硬塞进几何"这整类方案共同
#               的自由度矛盾(见聊天记录里的详细推导)。
#   geom_mode = "greedy"       → make_reg3_graph_geometric_greedy
#               拓扑真正随机(不像jitter拓扑锁死在蜂窝格上): 先随机撒点,
#               按距离从近到远贪心连边直到每个原子度数=3。综合了各家方法的
#               优点(详见函数文档字符串), 是从多篇论文里取长补短拼出来的
#               新方案, 不是照搬任何一篇。
#   geom_mode = "force3d"      → make_reg3_weighted_graph_force_3d
#               force的legacy k_mode搬到三维空间(自由度2n→3n, 用球体密堆积
#               公式撒点), 逻辑完全一样, 只是多一个维度去同时满足边近/非边
#               远这两类约束。
#   geom_mode = "greedy3d"     → make_reg3_graph_geometric_greedy_3d
#               greedy搬到三维空间, 同上。
#
# [已移除] geom_mode="target"(先指定目标权重, 再嵌入去凑位置): 实测在
# 随机3-regular拓扑上, 残差大到足以让"实际权重"和"最初目标权重"变成
# **负相关**(方向反了, 不只是不精确), 且DOF分析证明这个问题只会随n增大
# 而变严重, 不是能靠调参数救回来的, 已删除。k_mode="local"(论文Eq.8)同样
# 因为权重比值失控已删除, k_mode 现在只剩 global/legacy。
#
# [已移除] "geometric"(蜂窝格, 无抖动)选项: 权重必然全部相等, 本质上退化
# 成无权图 —— 已经独立成通用的无权图类型, 见 make_graph 里的
# graph_type="honeycomb"。加了抖动的版本(geom_mode="jitter")才有意义
# 用在这个"加权"场景下。
#
# 例如:
#   julia r3g_sge.jl gnn_qaoa_weighted 10 5 42 force 12.0 legacy
#   julia r3g_sge.jl gnn_qaoa_weighted 10 5 42 jitter 14.0 0.15
#   julia r3g_sge.jl gnn_qaoa_weighted 10 5 42 greedy 14.0
#   julia r3g_sge.jl gnn_qaoa_weighted 10 5 42 force3d 12.0
#   julia r3g_sge.jl gnn_qaoa_weighted 10 5 42 greedy3d 14.0
# ══════════════════════════════════════════════════════════════════════════

"""p=1→p_max, cold_start/tqa_init/gnn_warmstart 三条件完整对比 —— 从
   gnn_qaoa_weighted 抽出来的可复用版本, 让新的"一条龙"模式(预训练+QAOA)
   跟原来的 gnn_qaoa_weighted 共用同一套逻辑, 不用复制一份容易出岔子。
   内部通过 pmap 把每个 (p_val, condition, seed) 任务分发出去, 只要调用方
   所在的模式在 _compute_modes 里(会触发 addprocs), 这里就能自动利用到
   集群/多核 worker, 不需要额外改代码。"""
function run_qaoa_warmstart_comparison_weighted(atoms, edges, weights, n_atoms, n_edges, total_w,
                                                 best_p_gnn, gnn_ar, label, p_max::Int, seed::Int;
                                                 cmax_max_n::Int=24, verbose::Bool=true,
                                                 n_bo_seeds::Int=length(SEEDS))
    CONDITIONS = ["cold_start", "tqa_init", "gnn_warmstart"]
    # 主指标用 ⟨C⟩/C_max, 参考列保留 ⟨C⟩/Σw
    chain_ar   = Dict(c => Dict{Int,Float64}() for c in CONDITIONS)   # /C_max (主)
    chain_mean = Dict(c => Dict{Int,Float64}() for c in CONDITIONS)
    chain_std  = Dict(c => Dict{Int,Float64}() for c in CONDITIONS)
    chain_arw  = Dict(c => Dict{Int,Float64}() for c in CONDITIONS)   # /Σw (参考)
    prev_best  = Dict{String,Any}(c => nothing for c in CONDITIONS)   # [FIX] 显式Any类型, 否则Julia会把值类型锁死成Nothing

    # ── C_max: 暴力枚举真实最大割, 作为近似比的分母 ──────────────────────────
    # 对齐 Graham et al. (PRX Quantum / arXiv:2112.14589) 的定义:
    #   R_a = (1/S_max) Σ_i p_i S_i
    # 分子跟我们原来一样(整个测量分布上的期望值), 分母是**真实最大割**而不是
    # 总权重 Σw。为什么这个更合理: 任何含奇环的图都不可能切开所有边, 所以
    # Σw 是一个永远达不到的上界, 用它做分母的指标永远到不了1, 没法判断"还剩
    # 多少提升空间"; 除以 C_max 才是组合优化里近似比的标准定义, 也才能跟文献
    # 直接比(Graham 那篇给的理想无噪声理论值: 4比特T型图 p=1,2,3 分别是
    # 0.772/0.934/1.0 —— 这也说明有限p的天花板是QAOA的内在性质)。
    # [限制] 暴力枚举是 2^n, n 超过 cmax_max_n(默认24)就跳过, 退回只报 Σw
    # 归一化的数字。大规模时要报标准近似比, 得改用上界或好的启发式解。
    local cmax, has_cmax
    if n_atoms <= cmax_max_n
        verbose && print("  暴力枚举真实最大割 C_max (2^$(n_atoms) 种配置)... "); flush(stdout)
        t_bf = time()
        cmax = brute_force_maxcut(edges, weights, n_atoms)
        has_cmax = true
        verbose && @printf("C_max=%.4f  (C_max/Σw=%.4f, 这就是 Σw 归一化下的真实天花板)  用时%.1fs\n",
                cmax, cmax/total_w, time()-t_bf)
    else
        cmax = total_w; has_cmax = false
        println("  n_atoms=$(n_atoms) > $(cmax_max_n), 跳过暴力枚举, 主指标退回 Σw 归一化")
    end
    denom = has_cmax ? cmax : total_w
    for p_val in 1:p_max
        dim    = 2 * p_val
        lo     = zeros(dim)
        hi     = vcat([[3π, π] for _ in 1:p_val]...)
        # [FIX v2] 上一版把 budget 改成 100*dim + 20*dim²(p=3 时 1320), 结果
        # p=3 跑不完被中断。原因是我低估了代价的增长阶数: BO 每一步都要重建
        # 一次 GP, 核矩阵 N×N 的 Cholesky 是 O(N³), 而 N 随迭代从 n_init 涨到
        # budget —— 累计代价约 ~budget⁴, 不是线性。p=3 从 600 涨到 1320,
        # GP 累计代价涨了约 23 倍。
        # 而且加 budget 的动机本身就不充分: 之前判断的"高p时BO失效"经复查是
        # 过度解读(种子间标准差变大的是**均值**, 我们实际报的 best 一直稳定,
        # p=3 有4~5个种子都收敛到同一个 0.7994)。既然故障没被确认, 就不值得
        # 付 20 倍代价。这里回到接近原来的线性预算, 只做温和加码(+5*dim),
        # 高p靠已有的 20 个种子取最好来兜底。
        budget = max(200, BO_BUDGET_A*dim + BO_BUDGET_B*dim)
        n_init = max(20,    8 * dim)
        delta_t = 0.75
        tqa_angles = vcat([[delta_t*(k/p_val)*π, delta_t*(1-k/p_val)*π/2]
                           for k in 1:p_val]...)

        if verbose
            println("\n" * "─"^68)
            println("p=$(p_val)  dim=$(dim)D  budget=$(budget)  n_init=$(n_init)")
            println("─"^68)
        end
        t_layer = time()

        for cond in CONDITIONS
            println("\n  ── p=$(p_val) $(cond) (共 $(n_bo_seeds) 个BO种子) ──")
            flush(stdout)

            task_params = [(task_id    = tid,
                            p_val      = p_val,
                            budget     = budget,
                            n_init     = n_init,
                            cond       = cond,
                            angle_hint = cond == "tqa_init" ? tqa_angles : nothing,
                            prev_hint  = prev_best[cond],
                            t_start    = time())
                           for tid in 1:n_bo_seeds]

            cond_results = pmap(task_params) do par
                sd = SEEDS_W[((par.task_id-1) % length(SEEDS_W)) + 1]
                Random.seed!(sd)
                cnd = par.cond

                cache = Dict{Vector{Float64},Float64}()
                function ev!(x)
                    xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
                    haskey(cache, key) && return cache[key]
                    try
                        lv = if cnd == "gnn_warmstart"
                            loss_fn_warmstart_weighted(xs, best_p_gnn, atoms, edges, weights, n_atoms)
                        else
                            loss_fn_weighted(xs, atoms, edges, weights, n_atoms)
                        end
                        v = isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
                        cache[key] = v; return v
                    catch; return 1000.0; end
                end

                halton_pts = halton_samples(par.n_init, lo, hi; skip=(sd*7)%100)
                init_xs = Vector{Vector{Float64}}(undef, par.n_init)
                if cnd == "tqa_init"
                    n_tqa = max(3, par.n_init÷3)
                    for i in 1:n_tqa
                        sc = 0.05 + 0.10*(i-1)/n_tqa
                        init_xs[i] = clamp.(par.angle_hint .+ randn(dim).*sc.*(hi.-lo), lo, hi)
                    end
                    for i in (n_tqa+1):par.n_init
                        init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                    end
                elseif !isnothing(par.prev_hint) && length(par.prev_hint)==dim
                    n_h = max(3, par.n_init÷3)
                    for i in 1:n_h
                        sc = 0.05 + 0.10*(i-1)/n_h
                        init_xs[i] = clamp.(par.prev_hint.+randn(dim).*sc.*(hi.-lo), lo, hi)
                    end
                    for i in (n_h+1):par.n_init
                        init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                    end
                else
                    for i in 1:par.n_init
                        init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                    end
                end
                init_ys = Float64[ev!(x) for x in init_xs]
                X=hcat(init_xs...); y=copy(init_ys)
                by=minimum(init_ys); bx=init_xs[argmin(init_ys)]

                Q_g=2; Q_c=2
                # [FIX] 难度档位的判据原来是 dim>=8, 而 dim=2p, 所以 p=3(dim=6)
                # 落在低维档 —— 但 p=3 恰恰是我们实际跑的最难的情况, 却在用跟
                # p=1 一样的配置(候选池400、NelderMead只迭代100次、精修尺度只有
                # 3档粗网格)。这个阈值应该是当初为 Q3 那套实验定的, 后来 p 的范围
                # 变了没跟着改。改成 dim>=BO_DIM_THRESH 让 p=3 吃到高维档配置。
                # [注意] 这改变了 p=3 的数值结果, 已有的 p=3 checkpoint 会与新
                # 配置不一致 —— 想要一致的数据集, 删掉 ckpt 目录重跑。
                n_cand       = dim>=BO_DIM_THRESH ? 800 : 400
                map_every    = dim>=BO_DIM_THRESH ? 12 : 18
                phase1_end   = dim>=BO_DIM_THRESH ? 0.45 : 0.30
                phase2_end   = dim>=BO_DIM_THRESH ? 0.78 : 0.68
                nr_hp        = dim>=BO_DIM_THRESH ? 4 : 2
                stag_limit   = dim>=BO_DIM_THRESH ? 25 : 20
                refine_every = 15
                nm_iters     = dim>=BO_DIM_THRESH ? 200 : 100
                refine_scales = dim>=BO_DIM_THRESH ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
                θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp); stag=0

                for iter in 1:(par.budget-par.n_init)
                    t=iter/(par.budget-par.n_init)
                    iter>1 && iter%map_every==0 &&
                        (θ_gp=map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp))
                    gp=build_spectral_gp(X,y,θ_gp,Q_g,Q_c,dim)
                    C=lo.+rand(dim,n_cand).*(hi.-lo)
                    if t<phase1_end
                        xn=C[:,ts_acq(gp,C;ns=10)]
                    elseif t<phase2_end
                        ξ=0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
                        μ,σ=gp_pred(gp,C); xn=C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
                    else
                        β=max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
                        μ,σ=gp_pred(gp,C); xn=C[:,argmin(lcb_acq(μ,σ;β=β))]
                    end
                    yn=ev!(xn); X=hcat(X,xn); push!(y,yn)
                    if yn<by; by=yn; bx=copy(xn); stag=0
                    else; t>=phase1_end && (stag+=1); end
                    if stag>=stag_limit
                        for _ in 1:2
                            xr=lo.+rand(dim).*(hi.-lo); yr=ev!(xr)
                            X=hcat(X,xr); push!(y,yr); yr<by && (by=yr;bx=copy(xr))
                        end; stag=0
                    end
                    if iter%refine_every==0 && t>=phase1_end
                        for sc in refine_scales
                            xs0=clamp.(bx.+randn(dim).*sc.*(hi.-lo),lo.+1e-6,hi.-1e-6)
                            r=optimize(x->ev!(clamp.(x,lo,hi)),xs0,NelderMead(),
                                       Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                            yr=Optim.minimum(r); xr=clamp.(Optim.minimizer(r),lo,hi)
                            yr<by-1e-6 && (by=yr;bx=xr;stag=0;X=hcat(X,xr);push!(y,yr))
                        end
                    end
                end
                ar   = -by/denom      # 主指标: /C_max (n太大退不回来时 denom=Σw)
                ar_w = -by/total_w    # 参考: /Σw
                # [FIX] 这一行不再受 verbose 控制。上一版为了减少输出把它关了,
                # 结果批量模式下要等一整个 (p,condition) 的 20 个 BO 任务全部跑完
                # 才有第一行输出(p=1 就是 4200 次量子演化), 看起来跟卡死一样 ——
                # 实测被误判过两次。逐任务打印是能拿到的最细粒度, 宁可日志长一点。
                println("    task=$(par.task_id) [$(cnd)] p=$(p_val) ar=$(round(ar,digits=4))  (/Σw=$(round(ar_w,digits=4)))")
                (par.task_id, p_val, sd, cnd, by, bx, ar, ar_w)
            end

            ars  = [r[7] for r in cond_results]
            arws = [r[8] for r in cond_results]
            chain_ar[cond][p_val]   = maximum(ars)
            chain_mean[cond][p_val] = mean(ars)
            chain_std[cond][p_val]  = std(ars)
            chain_arw[cond][p_val]  = maximum(arws)
            # 这一行即使 verbose=false 也打印 —— 批量模式下如果全静音, 一个实例
            # 要跑几十分钟才有输出, 看起来跟卡死一模一样(实测被误判过)。
            @printf("    [%s] p=%d: best=%.4f  mean=%.4f±%.4f  (/Σw best=%.4f)  [%.1f分钟]\n",
                    cond, p_val, chain_ar[cond][p_val], chain_mean[cond][p_val],
                    chain_std[cond][p_val], chain_arw[cond][p_val], (time()-t_layer)/60)
            flush(stdout)

            best_r = argmax(r->r[7], cond_results)
            prev_best[cond] = best_r[6]
        end
        # 报告本层耗时 + 预告下一层大致要多久, 免得跑到一半才发现来不及。
        # 代价主要来自 BO 内部每步重建 GP(Cholesky O(N³), N 从 n_init 涨到
        # budget), 累计约 ~budget⁴, 所以按 budget 的四次方比例外推。
        el = time() - t_layer
        if p_val < p_max
            b_next = max(200, 100*(2*(p_val+1)) + 5*(2*(p_val+1)))
            verbose && @printf("  [耗时] p=%d 用了 %.1f 分钟; 按 budget⁴ 外推, p=%d 大约需要 %.1f 分钟\n",
                    p_val, el/60, p_val+1, el/60 * (b_next/budget)^4)
        else
            verbose && @printf("  [耗时] p=%d 用了 %.1f 分钟\n", p_val, el/60)
        end
    end

    mkpath(directoryBasis)
    save_tag = "init_compare_weighted_reg3w_n$(n_atoms)_seed$(seed)"
    serialize(joinpath(directoryBasis, string(save_tag, ".jls")),
              (chain_ar, chain_mean, chain_std, chain_arw, CONDITIONS, label,
               n_edges, weights, cmax, has_cmax, total_w))
    verbose && println("\n✓ Saved → " * joinpath(directoryBasis, save_tag * ".jls"))
    if verbose

        println("\n\n" * "═"^70)
        println("INITIALISATION COMPARISON (加权) — FINAL SUMMARY")
        println("$(label)  $(length(SEEDS)) seeds")
        if has_cmax
            @printf("主指标 = ⟨C⟩/C_max (对齐 Graham et al. arXiv:2112.14589 的 R_a 定义)\n")
            @printf("C_max=%.4f   Σw=%.4f   C_max/Σw=%.4f\n", cmax, total_w, cmax/total_w)
        else
            println("主指标 = ⟨C⟩/Σw (n_atoms 太大, 没做暴力枚举, 拿不到 C_max)")
        end
        println("═"^70)
        println("  p │ dim │   cold_start        │   tqa_init          │   gnn_warmstart")
        println("    │     │  best   mean±std    │  best   mean±std    │  best   mean±std")
        println("  " * "─"^68)
        for p_val in 1:p_max
            @printf("  %d │  %2dD │ %.4f  %.4f±%.4f │ %.4f  %.4f±%.4f │ %.4f  %.4f±%.4f\n",
                p_val, 2*p_val,
                chain_ar["cold_start"][p_val],   chain_mean["cold_start"][p_val],   chain_std["cold_start"][p_val],
                chain_ar["tqa_init"][p_val],     chain_mean["tqa_init"][p_val],     chain_std["tqa_init"][p_val],
                chain_ar["gnn_warmstart"][p_val],chain_mean["gnn_warmstart"][p_val],chain_std["gnn_warmstart"][p_val])
        end
        if has_cmax
            println("  " * "─"^68)
            println("  [参考] 同样的结果按 Σw 归一化 (跟这次改动之前的旧数字可直接对比):")
            println("  p │ dim │ cold_start │ tqa_init │ gnn_warmstart   (都是 best)")
            for p_val in 1:p_max
                @printf("  %d │  %2dD │   %.4f   │  %.4f  │    %.4f\n",
                    p_val, 2*p_val, chain_arw["cold_start"][p_val],
                    chain_arw["tqa_init"][p_val], chain_arw["gnn_warmstart"][p_val])
            end
        end
        println("  " * "─"^68)
        if has_cmax
            @printf("  LGNN standalone (纯经典GNN离散解): /C_max=%.4f   /Σw=%.4f\n",
                    gnn_ar*total_w/cmax, gnn_ar)
        else
            @printf("  LGNN standalone (纯经典GNN离散解): /Σw=%.4f\n", gnn_ar)
        end
        println("═"^70)
    end
    return chain_ar, chain_mean, chain_std, chain_arw, cmax, has_cmax
end

if length(ARGS) >= 1 && ARGS[1] == "gnn_qaoa_weighted"
    n_at      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    p_max     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
    seed      = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 42
    geom_mode = length(ARGS) >= 5 ? ARGS[5] : "force"
    d_fixed   = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 14.0
    k_mode    = length(ARGS) >= 7 ? ARGS[7] : "legacy"           # force模式下用
    # [FIX] 原来这里不管 geom_mode 是什么, 只要 length(ARGS)>=7 就会同时尝试
    # 把 ARGS[7] 解析成 Float64(给jitter模式用)——force模式下 ARGS[7] 是
    # "legacy"/"global" 这种字符串, 解析Float64直接崩溃。改成只在 geom_mode
    # 确实是"jitter"时才去解析, 其它模式下给个不会用到的默认值即可。
    jitter_frac = (length(ARGS) >= 7 && geom_mode == "jitter") ? parse(Float64, ARGS[7]) : 0.15
    use_prior = length(ARGS) >= 8 ? ARGS[8] : "none"   # 预训练先验文件名, "none"=走原来的单图训练
    using Statistics, LinearAlgebra

    local edges, weights, atoms, stress_val
    unwanted_ratio = NaN
    weight_fidelity = NaN
    if geom_mode == "jitter"
        atoms, edges, weights, unwanted_ratio =
            make_honeycomb_graph_jittered(n_at, d_fixed; jitter_frac=jitter_frac, seed=seed)
        stress_val = 0.0   # 没有优化步骤, 这个概念不适用, 打0只是为了跟其它模式统一输出格式
    elseif geom_mode == "greedy"
        atoms, edges, weights, unwanted_ratio =
            make_reg3_graph_geometric_greedy(n_at, d_fixed, seed)
        stress_val = 0.0   # 同上, 没有优化步骤
    elseif geom_mode == "greedy3d"
        atoms, edges, weights, unwanted_ratio =
            make_reg3_graph_geometric_greedy_3d(n_at, d_fixed, seed)
        stress_val = 0.0
    elseif geom_mode == "force3d"
        atoms, edges, weights, stress_val, unwanted_ratio =
            make_reg3_weighted_graph_force_3d(n_at, seed; d_target=d_fixed)
    elseif geom_mode == "target3d"
        atoms, edges, weights, stress_val, unwanted_ratio, weight_fidelity =
            make_reg3_weighted_graph_target3d(n_at, seed)
    else
        atoms, edges, weights, stress_val, unwanted_ratio =
            make_reg3_weighted_graph_force(n_at, seed; k_mode=k_mode, d_target=d_fixed)
    end
    n_atoms = length(atoms); n_edges = length(edges)
    total_w = sum(weights)
    label = geom_mode == "force" ? "reg3-weighted(n=$(n_atoms),geom=force,k=$(k_mode))" :
                                    "reg3-weighted(n=$(n_atoms),geom=$(geom_mode))"

    println("\n" * "="^70)
    println("INITIALISATION COMPARISON (带权图): cold-start vs TQA vs GNN warm-start")
    println("Graph:  $(label)  |V|=$(n_atoms)  |E|=$(n_edges)  嵌入残差=$(round(stress_val,digits=4))")
    !isnan(unwanted_ratio) &&
        @printf("非边耦合/边耦合比值 (unwanted_ratio): %.6f  (越接近0越好)\n", unwanted_ratio)
    !isnan(weight_fidelity) &&
        @printf("目标权重 vs 实际权重 相关系数: %.4f  (越接近1说明嵌入越贴合原始目标)\n", weight_fidelity)
    println("权重范围: [$(round(minimum(weights),digits=3)), $(round(maximum(weights),digits=3))]  Σw=$(round(total_w,digits=3))")
    println("p:      1 → $(p_max)  ·  3 conditions  ·  $(length(SEEDS)) seeds each")
    println("="^70)

    # ── Step 1: 训练带权 LGNN (或加载缓存, 或直接用预训练先验做一次推理) ─────
    local gnn_params, best_p_gnn, gnn_ar
    if use_prior != "none"
        prior_file = joinpath(directoryBasis, use_prior)
        if !isfile(prior_file)
            println("找不到先验文件: $(prior_file)"); exit(1)
        end
        println("\n加载预训练先验 $(basename(prior_file)), 对这张图做一次前向推理(不训练)...")
        gnn_params = deserialize(prior_file)
        best_p_gnn, gnn_ar = gnn_infer_weighted(gnn_params, edges, weights, n_atoms)
        @printf("  LGNN 推理结果 (weighted, 真实离散cut): /Σw=%.4f  (下面会一并给出 /C_max)\n", gnn_ar)
    else
        cache_tag = geom_mode == "force" ? "$(geom_mode)_$(k_mode)" : geom_mode
        gnn_file = joinpath(directoryBasis, "gnn_reg3w_n$(n_atoms)_seed$(seed)_$(cache_tag).jls")
        if isfile(gnn_file)
            println("\nLoading pre-trained weighted LGNN from $(basename(gnn_file))")
            gnn_params, best_p_gnn, gnn_ar, _, _, _ = deserialize(gnn_file)
            @printf("  LGNN standalone ar (weighted) = %.4f\n", gnn_ar)
        else
            println("\nTraining LGNN (300 epochs, unsupervised WEIGHTED MaxCut QUBO)...")
            gnn_params, best_p_gnn, gnn_ar = train_lgnn_quick_weighted(edges, weights, n_atoms; n_epochs=300)
            @printf("  LGNN standalone ar (weighted) = %.4f\n", gnn_ar)
            mkpath(directoryBasis)
            serialize(gnn_file, (gnn_params, best_p_gnn, gnn_ar, n_atoms, n_edges, edges))
            println("  Saved LGNN → $(gnn_file)")
        end
    end

    # ── Step 2: p=1→p_max chain, 3 conditions (共用逻辑, 见 run_qaoa_warmstart_comparison_weighted) ──
    run_qaoa_warmstart_comparison_weighted(atoms, edges, weights, n_atoms, n_edges, total_w,
                                            best_p_gnn, gnn_ar, label, p_max, seed)
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── gnn_pretrain_and_qaoa_weighted: 一条命令完成"预训练 + QAOA warmstart" ──
# 用 target3d 作为图构造方案(三轮实测下来综合表现最好的主力方案: 权重比值
# 接近最初设定的目标、unwanted_ratio 跟 force3d-legacy 一个量级、目标权重
# 相关系数≈0.99, 见聊天记录), 不是 force3d。
# 对应论文(Jain et al. 2022) Section 4.1 的训练集配置(训练图规模覆盖目标图
# 规模的若干倍, 混合多个规模), 但倍数范围收窄到 1x~3x, 不是论文原文的1x~5x
# ——实测发现n=48/60(5x那两档, 对n_atoms=12的测试图而言)的target3d嵌入
# 质量明显退化(非边最近距离跟边的目标距离区间开始重叠), 论文4.1节的1x~5x
# 是给理想化、没有真实物理嵌入代价的场景设计的, 我们这条流水线有真实嵌入
# 成本, 收窄到1x~2x是根据实测证据做的调整, 不是照搬论文数字。(论文原文还
# 用N_train=5000张图; 单机上5000张每张都要单独跑一次target3d嵌入优化, 不
# 现实, 这里给一个更实际的数字, 真的上集群跑可以调大)。
# 训练池用 build_graph_pool_weighted_target3d(权重跟测试时同源的target3d
# 嵌入, 不是独立随机权重就直接用, 修正train/test权重分布不匹配问题)。
# [FIX] 种子数不再是每个规模用同一个固定值——实测发现不同规模的干净比例
# 差异巨大(n=12约45%, n=24约20%, n=36约0%), 固定种子数会导致小规模样本
# 过多、大规模样本过少甚至为0(纯粹浪费算力)。现在改成"每个规模先探路
# 估计干净比例, 再按比例自适应补种子, 凑够 target_clean_per_size 张干净图"
# ——这个参数现在的含义是"每个规模想要多少张*干净*的图", 不是"每个规模
# 跑多少个种子", 语义变了。
#
# 权重范围固定用 wmin=0.5/wmax=1.0(比值2.0), 不是随手挑的, 是有实测数据
# 支撑的选择: 范围太窄(比值接近1)权重几乎均匀, 退化回"测不出加权行为"的
# 老问题; 范围太宽, 三维力场同时满足差异很大的逐边目标会明显变难, 嵌入
# 质量会跟着恶化(2D target模式失败的同一个根源, 三维只是把失控门槛推后,
# 不是没有上限)。wmin=0.5/wmax=1.0 是我们已经实测验证过的配置(n=12,
# seed=7: unwanted_ratio=0.084, 目标权重相关系数=0.9898), 不需要再扫描。
#
# 池构造和QAOA角度搜索都通过pmap分发, 这个模式已经加进 _compute_modes,
# 在SGE/SLURM/PBS集群或本地多核都能自动利用到worker, 不需要额外配置。
#
# 运行方式:
#   julia r3g_sge.jl gnn_pretrain_and_qaoa_weighted <n_atoms> [p_max] [seed] [n_epochs] [target_clean_per_size]
# 例如:
#   julia r3g_sge.jl gnn_pretrain_and_qaoa_weighted 12 5 7 1500 20
#   (每个规模想凑够20张干净图, 具体要跑多少个种子由代码自适应决定)
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "gnn_pretrain_and_qaoa_weighted"
    n_at                   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    p_max                  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
    seed                   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 42
    n_epochs               = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 8000
    target_clean_per_size  = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20
    optimizer              = length(ARGS) >= 7 ? ARGS[7] : "zygote"   # "zygote"(默认) 或 "spsa"
    wmin_fixed, wmax_fixed = 0.5, 1.0   # 已验证过的配置, 见上面注释
    using Statistics, LinearAlgebra
    # Zygote 按需自动检测/安装, 装不上就干净地退回 SPSA(而不是直接崩)
    if optimizer == "zygote" && !ensure_zygote()
        println("  → 退回 SPSA 优化器继续(结果会存成 spsa 的缓存名, 不会跟 zygote 的混淆)")
        optimizer = "spsa"
    end

    println("\n" * "="^70)
    println("GNN 预训练 + QAOA warmstart 一条龙 (target3d, wmin=$(wmin_fixed) wmax=$(wmax_fixed))")
    println("测试图: n_atoms=$(n_at)  p_max=$(p_max)  seed=$(seed)  优化器=$(optimizer)")
    println("训练集规模覆盖 $(n_at)~$(2*n_at) 之间所有偶数规模, 每个规模自适应补种子凑够 $(target_clean_per_size) 张干净图")
    LGNN_SEPARATE_LINE_WEIGHTS &&
        println("[注意] LGNN_SEPARATE_LINE_WEIGHTS=true, 用的是修复后的架构(线图项独立权重), 跟默认架构的先验不通用")
    println("="^70)

    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean$(target_clean_per_size)_$(optimizer)" *
                 (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)

    if !isfile(prior_file)
        println("没有找到匹配这组配置的先验, 现在构建训练池并预训练...")
        pool = build_graph_pool_weighted_target3d(n_at, target_clean_per_size; wmin=wmin_fixed, wmax=wmax_fixed)
        println("训练池共 $(length(pool)) 张图 (规模范围 $(minimum(g.n_atoms for g in pool))~$(maximum(g.n_atoms for g in pool)))")
        params, best_avg = if optimizer == "zygote"
            train_lgnn_pretrain_weighted_zygote(pool; n_epochs=n_epochs, min_epochs=min(2000, n_epochs), batch_size=24)
        else
            train_lgnn_pretrain_weighted(pool; n_epochs=n_epochs, batch_size=24)
        end
        mkpath(directoryBasis)
        serialize(prior_file, params)
        @printf("✓ 预训练完成, 训练池平均真实离散cut=%.4f, 已缓存 → %s\n", best_avg, prior_tag)
    else
        println("找到已缓存的先验, 直接复用 → $(prior_tag)")
    end

    gnn_params = deserialize(prior_file)

    atoms, edges, weights, stress_val, unwanted_ratio, fidelity =
        make_reg3_weighted_graph_target3d(n_at, seed; wmin=wmin_fixed, wmax=wmax_fixed)
    n_atoms = length(atoms); n_edges = length(edges); total_w = sum(weights)
    best_p_gnn, gnn_ar = gnn_infer_weighted(gnn_params, edges, weights, n_atoms)
    label = "reg3-weighted(n=$(n_atoms),geom=target3d)"

    println("\nGraph:  $(label)  |V|=$(n_atoms)  |E|=$(n_edges)  嵌入残差=$(round(stress_val,digits=4))")
    @printf("非边耦合/边耦合比值 (unwanted_ratio): %.6f\n", unwanted_ratio)
    @printf("目标权重 vs 实际权重 相关系数: %.4f\n", fidelity)
    @printf("GNN 推理结果 (weighted, 真实离散cut): /Σw=%.4f  (下面会一并给出 /C_max)\n", gnn_ar)

    run_qaoa_warmstart_comparison_weighted(atoms, edges, weights, n_atoms, n_edges, total_w,
                                            best_p_gnn, gnn_ar, label, p_max, seed)
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── [QNG] 量子自然梯度局部精修 ─────────────────────────────────────────────
# 依据: Stokes, Izaac, Killoran, Carleo, "Quantum Natural Gradient",
#       Quantum 4, 269 (2020), arXiv:1909.02108。
#   更新规则 (原文 Eq.12/13):  g(θ_t)(θ_{t+1} - θ_t) = -η ∇L(θ_t)
#   g_ij = Re G_ij,  G_ij = <∂_iψ|∂_jψ> - <∂_iψ|ψ><ψ|∂_jψ>   (原文 Eq.7, QGT)
#   原文按建议直接解线性方程组而不显式求伪逆; 这里加 Tikhonov 正则 λI 防奇异。
#
# 跟原文的三处差别(均为模拟器场景下的必要适配, 论文里要写明):
#  (1) 度量张量用**完整** QGT, 不做块对角近似。原文的块对角近似(Eq.18)是为了
#      在量子硬件上只用 L 次测量估计; 它还要求同层生成元对易(Eq.19-20之后的假设),
#      而我们的里德堡模拟脉冲每段都是 Ω·X + Δ·n + V·nn 的整体演化, 这个假设不
#      成立, 块对角近似本来也用不了。模拟器能直接拿到 |ψ(θ)>, 所以用中心差分
#      ∂_iψ ≈ [ψ(θ+h e_i) - ψ(θ-h e_i)]/2h 算全部 d×d 分量, 比原文近似更精确。
#      梯度复用同一批态: ∂_iL = [L(θ+h e_i) - L(θ-h e_i)]/2h。每次迭代 2d+1 次
#      演化(原文硬件方案是 2d+L 次, 量级相同)。
#  (2) 参数是脉冲**时长**, 有盒约束 [lo,hi]: 边界上改用单侧差分; 边界上梯度指向
#      外侧的坐标视为"被约束住", 从线性方程组里剔除(投影自然梯度)。
#  (3) 步长: 原文 η 是手调常数。这里用无噪声模拟器的标准做法 —— 回溯线搜索
#      (Armijo 条件) + 自适应 η(成功则加倍, 否则取接受值), 外加信赖域上限:
#      单步在任一坐标上不超过量程的 QNG_STEP_FRAC。这样 QNG 没有需要调的超参数,
#      跟同样"免调参"的 NelderMead 比才公平。
#
# 评估预算: 每次调用最多 max_evals 次量子演化(含差分与线搜索), 默认取
#   QNG_EVAL_FACTOR × (NelderMead 的迭代上限) —— NelderMead 平均每次迭代约
#   1~2 次函数评估, 1.5 倍是名义上的等预算。实际消耗两边都逐阶段记进 trace CSV
#   (n_evals 列), 比较时以实测评估次数为准, 不以名义值为准。
# ══════════════════════════════════════════════════════════════════════════
# ── 参数盒 (每层 cost 时长 ∈ [0, 3π·s], mixer 时长 ∈ [0, π·s]) ───────────────
# 里德堡模拟哈密顿量的耦合不是整数、失谐全程开着, 所以角度空间**不是周期的**,
# "整个参数空间"是无界的 —— [0,3π]×[0,π] 只是我们选的搜索盒。想检验盒子是否
# 限制了结果, 设 QAOA_BOX_SCALE=2 (盒子每边放大 2 倍) 重跑对比即可; 默认 1 与
# 旧版完全相同。≠1 时 checkpoint 目录名和输出文件名自动加 _box<s> 后缀。
# (只作用于 gnn_batch_qaoa_weighted 用到的 bo / layerwise / beam / sweep / hybrid 链。)
_box_scale = parse(Float64, get(ENV, "QAOA_BOX_SCALE", "1.0"))
_box_scale > 0 || error("QAOA_BOX_SCALE 必须为正数, 收到 $(_box_scale)")
@everywhere const QAOA_BOX_SCALE = $_box_scale
# ── 混合层 Rabi 幅度 Ωt (Bauer et al. 方法二里取 Ω=2; QAOA_OMEGA_T 可改) ──────────
#   混合层每个原子绕"合成轴"转: 横场 Ωt/2 把轴拉向 x, 一直开着的 ZZ(每原子约 Σ_j J_ij ≈ 2.3)
#   把轴拉向 z。Ωt 越大混合层越接近理想 X 旋转。转角 = Ωt·β, 所以 β 的盒子按 2/Ωt 缩放,
#   使转角范围始终是 [0, 2π·s](Ωt=2 时与旧版逐比特相同)。
#   OMEGA_T_REF 是 Ref: omega_scan 模式在 worker 上逐任务改写它; 其它模式只读。
#   ≠2 时: transfer_bo 的目录名加 _om<Ωt>; 其它模式的 checkpoint 名**不含** Ωt —— 会打印警告。
#   (QAOA_OMEGA_T / OMEGA_T_REF 定义在文件开头 —— 所有模式的演化函数都要用到)
@everywhere const BOX_HI = (3π * QAOA_BOX_SCALE, π * QAOA_BOX_SCALE * 2.0 / QAOA_OMEGA_T)
if QAOA_OMEGA_T != 2.0
    println("混合层幅度 Ωt = $(QAOA_OMEGA_T) (默认 2): β 盒子 → [0,$(round(BOX_HI[2], digits=3))] (转角上限仍为 2π·$(QAOA_BOX_SCALE))")
    _current_mode in ("transfer_bo", "omega_scan", "gnn_batch_qaoa_weighted", "noise_eval", "noise_selftest") ||
        println("⚠ 警告: 模式 $(_current_mode) 的 checkpoint 目录名不含 Ωt, 与 Ωt=2 的旧结果会混在一起 —— 请换输出目录或只用 transfer_bo / omega_scan")
end
QAOA_BOX_SCALE != 1.0 && println("参数盒放大: QAOA_BOX_SCALE=$(QAOA_BOX_SCALE) → 每层 [0,$(round(BOX_HI[1],digits=3))]×[0,$(round(BOX_HI[2],digits=3))]")

@everywhere begin
const QNG_FD_H        = 1e-3    # 有限差分步长(时长单位 μs; 角度量程 ~1..10)
const QNG_LAM_REL     = 1e-4    # Tikhonov 正则, 相对于 tr(g)/d
const QNG_STEP_FRAC   = 0.10    # 信赖域: 单步每个坐标最多移动量程的 10%
const QNG_EVAL_FACTOR = 1.5     # QNG 评估上限 = 1.5 × NelderMead 迭代上限
const QNG_MAX_BT      = 6       # 线搜索最多回溯 6 次(步长最小缩到 1/64)
# 以上任何一项改动都会改变数值结果 → 拼进 checkpoint 目录名, 旧结果自动失效
const QNG_CFG_TAG     = "qngH$(QNG_FD_H)L$(QNG_LAM_REL)S$(QNG_STEP_FRAC)E$(QNG_EVAL_FACTOR)B$(QNG_MAX_BT)"

"""量子自然梯度下降(盒约束, 投影 + 回溯线搜索)。
   svf(x) -> (loss, ψ::Vector{ComplexF64}); 失败时返回 (loss, nothing)。
   返回 (x, f, nev, iters, reason); reason ∈ :converged / :budget / :linesearch / :state_fail。
   只会返回不劣于起点的点(每一步都要求严格下降)。"""
function qng_minimize(svf, x0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                      max_evals::Int=150, h::Float64=QNG_FD_H, lam_rel::Float64=QNG_LAM_REL,
                      step_frac::Float64=QNG_STEP_FRAC, max_bt::Int=QNG_MAX_BT,
                      ftol::Float64=1e-10, gtol::Float64=1e-8)
    d   = length(x0)
    rng = hi .- lo
    x   = clamp.(copy(x0), lo, hi)
    f, ψ = svf(x); nev = 1
    ψ === nothing && return (x=x, f=f, nev=nev, iters=0, reason=:state_fail)
    η = Inf                      # 第一步只受信赖域上限约束, 之后自适应
    it = 0; n_small = 0; reason = :budget
    while true
        # 一次迭代至少要 2d 次差分 + 1 次线搜索, 不够就停
        if nev + 2d + 1 > max_evals
            reason = :budget; break
        end
        it += 1
        grad = zeros(d)
        dψ   = zeros(ComplexF64, length(ψ), d)
        ok   = true
        for i in 1:d
            xp = copy(x); xm = copy(x)
            xp[i] = min(x[i] + h, hi[i]); xm[i] = max(x[i] - h, lo[i])   # 边界处自动变单侧差分
            hh = xp[i] - xm[i]
            hh > 0 || continue                                          # 退化坐标(lo==hi)
            if xp[i] == x[i]; fp, ψp = f, ψ; else; fp, ψp = svf(xp); nev += 1; end
            if xm[i] == x[i]; fm, ψm = f, ψ; else; fm, ψm = svf(xm); nev += 1; end
            if ψp === nothing || ψm === nothing
                ok = false; break
            end
            grad[i]   = (fp - fm) / hh
            dψ[:, i] .= (ψp .- ψm) ./ hh
        end
        if !ok
            reason = :state_fail; break
        end
        # Fubini–Study 度量 = QGT 实部 (Stokes et al. Eq.7/28)
        a = dψ' * ψ                                  # a_i = <∂_iψ|ψ>
        G = real.(dψ' * dψ .- a * a')                # (a*a')_ij = <∂_iψ|ψ><ψ|∂_jψ>
        G = (G .+ G') ./ 2
        # 投影: 贴在边界上且梯度指向外侧的坐标冻结
        F = [i for i in 1:d if rng[i] > 0 &&
                               !((x[i] <= lo[i] + 1e-12 && grad[i] > 0) ||
                                 (x[i] >= hi[i] - 1e-12 && grad[i] < 0))]
        if isempty(F) || norm(grad[F]) < gtol
            reason = :converged; break
        end
        # 解 (g+λI)Δ = -∇L (不显式求逆)。度量会把各坐标耦合起来, 所以即使某个贴边
        # 坐标的梯度指向内侧, 自然梯度方向在它上面仍可能指向外侧 —— 那样步长会被
        # 边界卡成 0、线搜索白白失败(原型测试里实测出现过, |∇L|~1 时就停了)。
        # 处理: 把这类坐标也冻结, 在剩下的坐标上重新求解, 直到方向可行。
        local dir, gF
        while true
            gF = grad[F]; GF = G[F, F]
            λ  = lam_rel * max(tr(GF) / length(F), 1e-12)
            dir = try
                -((Symmetric(GF) + λ * I) \ gF)
            catch
                -gF
            end
            (all(isfinite, dir) && dot(dir, gF) < 0) || (dir = -gF)   # 兜底: 退回普通梯度方向
            blk = [F[j] for j in eachindex(F) if (x[F[j]] <= lo[F[j]] + 1e-12 && dir[j] < 0) ||
                                                 (x[F[j]] >= hi[F[j]] - 1e-12 && dir[j] > 0)]
            isempty(blk) && break
            F = setdiff(F, blk)
            isempty(F) && break
        end
        if isempty(F) || norm(grad[F]) < gtol
            reason = :converged; break
        end
        mx = maximum(abs.(dir) ./ rng[F])
        if !(mx > 0)
            reason = :converged; break
        end
        s_nom = min(η, step_frac / mx)
        # 走到边界为止(fraction-to-boundary): 不让 clamp 截断步子 —— 被截断的
        # 自然梯度步未必还是下降方向。碰到边界的坐标下一轮自动进入冻结判断。
        smax = Inf
        for j in eachindex(F)
            i = F[j]
            if dir[j] < 0
                smax = min(smax, (x[i] - lo[i]) / (-dir[j]))
            elseif dir[j] > 0
                smax = min(smax, (hi[i] - x[i]) / dir[j])
            end
        end
        hit_bound = smax < s_nom
        s = min(s_nom, smax)
        # 回溯线搜索 (Armijo, c=1e-4), 每次试探 1 次演化
        accepted = false; nbt = 0
        fn = f; ψn = ψ; xn = x
        for bt in 0:max_bt
            nev + 1 > max_evals && break
            xt = copy(x); xt[F] .+= s .* dir
            for i in 1:d                              # 贴边的坐标精确落在边界上
                abs(xt[i] - lo[i]) < 1e-10 && (xt[i] = lo[i])
                abs(xt[i] - hi[i]) < 1e-10 && (xt[i] = hi[i])
            end
            xt = clamp.(xt, lo, hi)
            ft, ψt = svf(xt); nev += 1; nbt = bt
            if ψt !== nothing && ft < f && ft <= f + 1e-4 * min(dot(grad, xt .- x), 0.0)
                accepted = true; fn = ft; ψn = ψt; xn = xt
                break
            end
            s /= 2
        end
        if !accepted
            reason = (nev + 1 > max_evals) ? :budget : :linesearch
            break
        end
        gain = f - fn
        x = xn; f = fn; ψ = ψn
        # 一次就成功 → 下一步敢走更远; 若这一步只是被边界截短, 不因此缩小 η
        η = nbt == 0 ? (hit_bound ? s_nom : 2s) : s
        if gain < ftol * max(1.0, abs(f))
            n_small += 1
            if n_small >= 3
                reason = :converged; break
            end
        else
            n_small = 0
        end
    end
    return (x=x, f=f, nev=nev, iters=it, reason=reason)
end
end # @everywhere begin (QNG)

# ══════════════════════════════════════════════════════════════════════════
# ── 二维块搜索: 固定其余角度, 只优化指定那一层的 (γ_k, β_k) ────────────────
# 这是逐层策略的原子操作, 束搜索和层间轮扫都调它, 保证两者用的是**同一套**
# GSM+CSM BO, 不会因为实现分叉而无法比较。
# 已经用 60×60 穷举网格验证过: 18/18 个二维子问题上, 这个 BO(300次评估)
# 精确达到穷举最优 —— 所以二维块这一层是可靠的, 剩下的问题都在"怎么组织
# 这些块"上, 而不是块内部。
#
# x_full : 完整的 2p 维角度向量(会被复制, 不修改原值)
# k      : 要重新优化第几层 (1..p)
# seed_x : 该层的起始猜测; 传 nothing 则用 (0,0) + Halton 初始设计
#          [关键] 初始设计里含 (0,0): 零时长的 cost/mixer 层就是恒等算符,
#          所以新建一层时能精确还原 p-1 层 ⇒ 保证 R_a 单调不降。
# 返回 (best_x_full, best_loss)
# ══════════════════════════════════════════════════════════════════════════
@everywhere function optimize_layer_2d(ev_full, x_full::Vector{Float64}, k::Int, p_val::Int,
                                        lo::Vector{Float64}, hi::Vector{Float64};
                                        budget::Int=300, seed_x=nothing,
                                        use_grid::Bool=false, n_grid::Int=40,
                                        refiner::Symbol=:nm, sv_full=nothing,
                                        qng_evals::Int=round(Int, QNG_EVAL_FACTOR*100))
    # refiner=:nm  → 循环内精修用 NelderMead(原方案, 默认, 数值与旧版逐比特一致)
    # refiner=:qng → 同样的起点、同样的触发时机, 换成量子自然梯度(qng_minimize);
    #                需要 sv_full(x) -> (loss, ψ) 提供末态波函数
    refiner == :qng && sv_full === nothing &&
        error("optimize_layer_2d: refiner=:qng 需要提供 sv_full (返回 (loss, ψ) 的函数)")
    lo2 = [0.0, 0.0]; hi2 = [BOX_HI[1], BOX_HI[2]]
    i1, i2 = 2k-1, 2k
    base = copy(x_full)
    ev2(u) = (xx = copy(base); xx[i1] = u[1]; xx[i2] = u[2]; ev_full(xx))

    local by, bu
    if use_grid
        by = Inf; bu = [0.0, 0.0]
        for γ in range(0.0, BOX_HI[1], length=n_grid), β in range(0.0, BOX_HI[2], length=n_grid)
            v = ev2([γ, β]); v < by && (by = v; bu = [γ, β])
        end
    else
        n_init2 = min(24, max(12, budget ÷ 8))
        halton2 = halton_samples(n_init2, lo2, hi2; skip=7)
        init_us = Vector{Vector{Float64}}([[0.0, 0.0]])          # 单调性保证
        seed_x !== nothing && push!(init_us, clamp.(seed_x, lo2, hi2))
        for gf in (0.25, 0.5, 0.75); push!(init_us, [gf*3π, 0.5π]); end
        for i in 1:max(0, n_init2 - length(init_us))
            push!(init_us, clamp.(halton2[i] .+ randn(2).*0.05.*(hi2.-lo2), lo2, hi2))
        end
        ys = Float64[ev2(u) for u in init_us]
        U = hcat(init_us...); y = copy(ys)
        by = minimum(ys); bu = init_us[argmin(ys)]

        Q_g=2; Q_c=2; D2=2
        xsc2 = hi2 .- lo2                               # [GP v2] 输入量程(v1 时被忽略)
        θ = map_spectral_hp(U,y,Q_g,Q_c,D2;nr=2,xscale=xsc2); stag=0
        for iter in 1:(budget - length(init_us))
            t = iter/max(1, budget - length(init_us))
            iter>1 && iter%18==0 && (θ = map_spectral_hp(U,y,Q_g,Q_c,D2;nr=2,θ_init=θ,xscale=xsc2))
            gp = build_spectral_gp(U,y,θ,Q_g,Q_c,D2; xscale=xsc2)
            C  = lo2 .+ rand(D2,400).*(hi2.-lo2)
            un = if t < 0.30
                C[:,ts_acq(gp,C;ns=10)]
            elseif t < 0.68
                ξ = 0.05*exp(-3.0*(t-0.30)/0.38)
                μ,σ = gp_pred(gp,C); C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
            else
                β_l = max(exp(-2.5*(t-0.68)/0.32),0.05)
                μ,σ = gp_pred(gp,C); C[:,argmin(lcb_acq(μ,σ;β=β_l))]
            end
            yn = ev2(un); U = hcat(U,un); push!(y,yn)
            if yn < by; by = yn; bu = copy(un); stag = 0
            else; t >= 0.30 && (stag += 1); end
            if stag >= 20
                for _ in 1:2
                    ur = lo2 .+ rand(D2).*(hi2.-lo2); yr = ev2(ur)
                    U = hcat(U,ur); push!(y,yr); yr < by && (by = yr; bu = copy(ur))
                end; stag = 0
            end
            if iter % 15 == 0 && t >= 0.30
                for sc in (0.05, 0.20, 0.50)
                    # [QNG 受控对比] 起点 u0 的随机抽取对两种精修器完全相同
                    # (同样的 randn 调用), 只有"从 u0 出发用哪种局部优化器"不同。
                    u0 = clamp.(bu .+ randn(D2).*sc.*(hi2.-lo2), lo2.+1e-6, hi2.-1e-6)
                    if refiner == :qng && sv_full !== nothing
                        sv2 = u -> (xx = copy(base); xx[i1] = u[1]; xx[i2] = u[2]; sv_full(xx))
                        rq  = qng_minimize(sv2, u0, lo2, hi2; max_evals=qng_evals)
                        yr  = rq.f; ur = rq.x
                    else
                        rr = optimize(u -> ev2(clamp.(u, lo2, hi2)), u0, NelderMead(),
                                      Optim.Options(iterations=100, f_reltol=1e-7))
                        yr = Optim.minimum(rr); ur = clamp.(Optim.minimizer(rr), lo2, hi2)
                    end
                    if yr < by - 1e-6
                        by = yr; bu = ur; stag = 0
                        U = hcat(U,bu); push!(y,yr)
                    end
                end
            end
        end
    end
    xb = copy(base); xb[i1] = bu[1]; xb[i2] = bu[2]
    return xb, by
end

# ══════════════════════════════════════════════════════════════════════════
# ── 束搜索 / 层间轮扫: 两种攻击"前缀约束"的策略 ───────────────────────────
# 背景(有实测证据): 逐层冻结把 2p 维问题拆成可靠求解的二维块 —— 60×60 穷举
# 网格验证过 18/18 个子问题上 BO 精确达到最优。但代价是**早期层一旦定死就
# 不能改**: 实例3 的 p=3, 穷举了新层全部 3600 个角度组合仍只能到 0.8914,
# 而联合BO 有 0.9029 —— 那个更优解要求前面几层是**不同的配置**。
# 加大精修扰动(0.06→0.50 倍量程)和加 INTERP 起点都试过, 完全无效, 说明这
# 不是"局部搜索半径不够", 是搜索空间被结构性切割。
#
# 两种解法, 机制不同:
#   beam  : **对冲** —— 每层保留 K 个互不相同的前缀, 不把宝押在一个上。
#           找不到"从未进过候选集"的前缀(若最优前缀在 p-1 层排第20, 束宽5
#           就漏了), 但天然并行, 适合 worker 多的场景。
#   sweep : **纠正** —— 加完新层后回头把前面每层依次重新优化(块坐标下降)。
#           不受候选集限制, 每个子问题仍是二维(BO 依然精确), 且不改变
#           best-of-K 的统计含义。代价是顺序执行, 不好并行。
# 两者都只接受改进 ⇒ 单调性 R_a(p) >= R_a(p-1) 依然由构造保证。
# ══════════════════════════════════════════════════════════════════════════
@everywhere function run_one_layer_strategy_chain(g, bo_seed::Int, p_max::Int;
                                                   strategy::Symbol=:sweep,
                                                   beam_width::Int=5, n_sweeps::Int=2,
                                                   n_refine_top::Int=2,
                                                   layer_budget::Int=300, sweep_budget::Int=100,
                                                   polish_iters::Int=300, bo_verbose::Int=0,
                                                   budget_scale::Float64=1.0,
                                                   refiner::Symbol=:nm)
    # refiner: 局部精修器。:nm = NelderMead(原方案); :qng = 量子自然梯度。
    # 两处精修(二维BO循环内精修 + 全维度联合精修)同时切换, 其余一切(BO、
    # 初始设计、随机种子、起点扰动、轮扫、预算)完全相同 —— 受控对比。
    refiner in (:nm, :qng) || error("未知的 refiner=$(refiner), 只支持 :nm / :qng")
    # [FIX] budget_scale 原来没传进这三个新策略, 导致 `冒烟测试`(=0.05)对
    # sweep/beam/hybrid **完全无效** —— 本该几秒的一步实际跑了 5~10 分钟。
    # 现在三处预算(新层BO、轮扫、全维度精修)全部按比例缩放。
    layer_budget  = max(30, round(Int, budget_scale * layer_budget))
    sweep_budget  = max(20, round(Int, budget_scale * sweep_budget))
    polish_iters  = max(30, round(Int, budget_scale * polish_iters))
    atoms, edges, weights = g.atoms, g.edges, g.weights
    n_atoms, total_w, denom = g.n_atoms, g.total_w, g.denom
    best_p_gnn = g.best_p_gnn

    CONDS = ["cold_start", "tqa_init", "gnn_warmstart"]
    out   = Dict(c => Dict{Int,Float64}() for c in CONDS)
    outw  = Dict(c => Dict{Int,Float64}() for c in CONDS)
    outt  = Dict(c => Dict{Int,Float64}() for c in CONDS)
    outx  = Dict(c => Dict{Int,Vector{Float64}}() for c in CONDS)   # 最优角度(查贴边用)
    # beam: 每个 condition 维护 K 条前缀; sweep: 只维护 1 条
    beams = Dict{String,Vector{Vector{Float64}}}(c => [Float64[]] for c in CONDS)
    empty!(W_LOG_BUF)
    t_chain = time()
    # 结构化追踪: 每个 (p, condition) 的每个阶段各一条记录, 含耗时、内存、
    # 评估次数、当前最优。最后随返回值带回主进程写成 CSV, 便于事后分析
    # "时间花在哪、内存涨到多少、哪一步没有产出改进"。
    trace = NamedTuple[]
    trec(stage, p_val, cnd, ar, t0, nev; note="") = push!(trace, (
        ts       = Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        worker   = myid(),
        inst     = g.inst,
        graph_seed = g.seed,
        bo_seed  = bo_seed,
        p        = p_val,
        condition= cnd,
        stage    = stage,
        ar_cmax  = ar,
        # [FIX] round(x, n) 是 Python 的写法, Julia 必须用 round(x, digits=n) ——
        # 两参数版本在 Julia 里是 round(Type, x), 所以传 Int 会 MethodError。
        elapsed_s= round(time()-t0, digits=2),
        rss_gb   = round(Sys.maxrss()/2^30, digits=3),
        heap_gb  = round(Base.gc_live_bytes()/2^30, digits=3),
        n_evals  = nev,
        refiner  = String(refiner),
        note     = note))
    # QNG 精修的评估上限(名义上与 NelderMead 等预算, 见 QNG_EVAL_FACTOR 说明)
    qng_evals_2d     = round(Int, QNG_EVAL_FACTOR * 100)
    qng_evals_polish = round(Int, QNG_EVAL_FACTOR * polish_iters)

    for p_val in 1:p_max
        dim = 2*p_val
        lo  = zeros(dim); hi = vcat([[BOX_HI[1], BOX_HI[2]] for _ in 1:p_val]...)
        delta_t = 0.75
        tqa = vcat([[delta_t*(k/p_val)*π, delta_t*(1-k/p_val)*π/2] for k in 1:p_val]...)

        for cnd in CONDS
            t_cond = time()
            tag = "[$(strategy) i$(g.inst)/s$(bo_seed)/p$(p_val)/$(cnd)]"
            Random.seed!(bo_seed*1000 + p_val*10 + findfirst(==(cnd), CONDS))

            cache = Dict{Vector{Float64},Float64}()
            # nsim: 实际做过的量子演化次数(= 硬件无关的"成本")。NelderMead 路径下
            # 它恰好等于缓存增长量(跟旧版 trace 的 n_evals 定义完全一致); QNG 需要
            # 波函数本身, 缓存里已有的点也得重新演化, 所以必须单独计数才公平。
            nsim = Ref(0)
            function ev(x)
                xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
                haskey(cache, key) && return cache[key]
                nsim[] += 1
                v = try
                    lv = cnd == "gnn_warmstart" ?
                         loss_fn_warmstart_weighted(xs, best_p_gnn, atoms, edges, weights, n_atoms) :
                         loss_fn_weighted(xs, atoms, edges, weights, n_atoms)
                    isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
                catch; 1000.0; end
                cache[key] = v; return v
            end
            # [QNG] 返回 (loss, ψ)。loss 的定义与 ev 完全相同(同样的截断), 并写回
            # 缓存, 之后 BO/轮扫碰到同一点直接复用。失败返回 (1000.0, nothing)。
            function sv(x)
                xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
                nsim[] += 1
                try
                    reg = cnd == "gnn_warmstart" ?
                          state_warmstart_weighted(xs, best_p_gnn, atoms, n_atoms) :
                          state_weighted(xs, atoms, n_atoms)
                    lv  = -maxcut_ev_weighted(reg, edges, weights)
                    v   = isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
                    cache[key] = v
                    return v, ComplexF64.(vec(copy(reg.state)))
                catch
                    cache[key] = 1000.0
                    return 1000.0, nothing
                end
            end
            sv_arg = refiner == :qng ? sv : nothing

            # ── 在每条现有前缀上扩展新层 ──────────────────────────────
            cands = Tuple{Vector{Float64},Float64}[]
            for (bi, head) in enumerate(beams[cnd])
                t0 = time(); n0 = nsim[]
                x0 = vcat(head, [0.0, 0.0])                      # 含 (0,0) ⇒ 单调
                sx = length(head) >= 2 ? head[end-1:end] : nothing
                xb, yb = optimize_layer_2d(ev, x0, p_val, p_val, lo, hi;
                                            budget=layer_budget, seed_x=sx,
                                            refiner=refiner, sv_full=sv_arg,
                                            qng_evals=qng_evals_2d)
                push!(cands, (xb, yb))
                trec("expand_beam$(bi)", p_val, cnd, -yb/denom, t0, nsim[]-n0)
            end

            # ── 轮扫(块坐标下降): 回头把前面每一层重新优化一遍 ──────────
            # 局部函数, sweep 和 hybrid 共用
            do_sweep(xb, yb, label) = begin
                for s in 1:n_sweeps, k in 1:(p_val-1)
                    t0 = time(); n0 = nsim[]; y_before = yb
                    xk, yk = optimize_layer_2d(ev, xb, k, p_val, lo, hi;
                                                budget=sweep_budget, seed_x=xb[2k-1:2k],
                                                refiner=refiner, sv_full=sv_arg,
                                                qng_evals=qng_evals_2d)
                    improved = yk < yb - 1e-9
                    if improved
                        yb = yk; xb = xk
                        bo_verbose > 0 && wlog(@sprintf("%s   %s 轮扫%d 层%d 改进 → %.4f",
                                                        tag, label, s, k, -yb/denom), :detail)
                    end
                    # 没有改进的轮扫也**如实记录** —— 它同样花了时间, 是判断
                    # "轮扫值不值"的关键数据(空结果本身就是结果)。
                    trec("sweep$(s)_layer$(k)$(label)", p_val, cnd, -yb/denom, t0,
                         nsim[]-n0; note = improved ? "改进 $(round(-(yk-y_before)/denom, digits=5))" : "无改进")
                end
                (xb, yb)
            end

            if strategy == :sweep && p_val >= 2
                xb, yb = cands[argmin(c[2] for c in cands)]
                cands = [do_sweep(xb, yb, "")]
            elseif strategy == :hybrid && p_val >= 2
                # [叠加设计] 束负责广度, 轮扫负责深度, 关键在**顺序**:
                #  · 只对得分靠前的 n_refine_top 条做轮扫 —— 全部都扫会让它们
                #    收敛到同一个坐标下降不动点, 束宽退化成 1, 白花算力;
                #  · 轮扫放在**重新排序之前** —— 一条暂时排第3的前缀可能扫完
                #    反超第1, 若先排序后砍掉, 它只是"还没被精修过"就被淘汰了,
                #    这恰恰违背了对冲的初衷。
                sort!(cands, by = t -> t[2])
                for i in 1:min(n_refine_top, length(cands))
                    cands[i] = do_sweep(cands[i][1], cands[i][2], "束$(i)")
                end
            end

            # ── 全维度联合精修(只接受改进) ────────────────────────────
            newbeam = Vector{Vector{Float64}}()
            scored  = Tuple{Vector{Float64},Float64}[]
            for (ci, (xb0, yb0)) in enumerate(cands)
                xb, yb = copy(xb0), yb0
                # [代价控制] 全维度精修是这里最贵的一步(每个起点 300 次
                # NelderMead 迭代)。beam/hybrid 下有 K 条候选, 全部做完整精修
                # 就是 K 倍开销。只给得分靠前的 n_refine_top 条完整精修(4档
                # 扰动), 靠后的只做 1 个起点的轻量精修 —— 它们本来就是"备胎",
                # 真要翻盘也是靠下一层的轮扫, 不是靠现在多扫几个起点。
                full_polish = (strategy != :beam && strategy != :hybrid) || ci <= n_refine_top
                scales = full_polish ? (0.05, 0.15, 0.30, 0.50) : ()
                starts = Vector{Vector{Float64}}([copy(xb)])
                for (k, sc) in enumerate(scales)
                    pert = [((i*7 + k*13 + p_val*29) % 100)/100 - 0.5 for i in 1:dim]
                    push!(starts, clamp.(xb .+ pert .* sc .* (hi.-lo), lo, hi))
                end
                cnd == "tqa_init" && full_polish && push!(starts, clamp.(tqa, lo, hi))
                t0 = time(); n0 = nsim[]; y_pre = yb
                qstat = String[]                         # 每个起点 QNG 的迭代数/停止原因
                for xs0 in starts
                    if refiner == :qng
                        rq = qng_minimize(sv, xs0, lo, hi; max_evals=qng_evals_polish)
                        yv = rq.f; xv = rq.x
                        push!(qstat, "$(rq.iters)it/$(rq.nev)ev/$(rq.reason)")
                    else
                        r  = optimize(x -> ev(clamp.(x, lo, hi)), xs0, NelderMead(),
                                      Optim.Options(iterations=polish_iters, f_reltol=1e-8))
                        yv = Optim.minimum(r); xv = clamp.(Optim.minimizer(r), lo, hi)
                    end
                    yv < yb - 1e-9 && (yb = yv; xb = xv)
                end
                # note 里不能有逗号(trace 是 CSV), QNG 各起点的统计用 ';' 分隔
                trec("polish_cand$(ci)", p_val, cnd, -yb/denom, t0, nsim[]-n0;
                     note = (yb < y_pre - 1e-9 ? "改进 $(round(-(yb-y_pre)/denom, digits=5))" : "无改进") *
                            (isempty(qstat) ? "" : " | qng " * join(qstat, ";")))
                push!(scored, (xb, yb))
            end

            sort!(scored, by = t -> t[2])
            best_x, best_y = scored[1]
            if strategy == :beam || strategy == :hybrid
                # 保留最好的 K 条**互不相同**的前缀(去重: 角度差异过小视为同一条)
                for (xb, _) in scored
                    isdup = any(nb -> maximum(abs.(nb .- xb)) < 1e-3, newbeam)
                    !isdup && length(newbeam) < beam_width && push!(newbeam, xb)
                end
                isempty(newbeam) && push!(newbeam, best_x)
                beams[cnd] = newbeam
            else
                beams[cnd] = [best_x]
            end

            out[cnd][p_val]  = -best_y/denom
            outw[cnd][p_val] = -best_y/total_w
            outt[cnd][p_val] = time() - t_cond
            outx[cnd][p_val] = copy(best_x)
            wlog(@sprintf("      · inst%d/p=%d %-14s ar=%.4f (/Σw=%.4f)  [%s+%s, %d条前缀]  [%d次演化]  [%.1f分钟]",
                          g.inst, p_val, cnd, out[cnd][p_val], outw[cnd][p_val],
                          strategy, refiner, length(beams[cnd]), nsim[], outt[cnd][p_val]/60), :summary)
        end
    end
    return (inst=g.inst, bo_seed=bo_seed, out=out, outw=outw, outt=outt,
            logbuf=copy(W_LOG_BUF), trace=trace, refiner=String(refiner), outx=outx)
end

# ══════════════════════════════════════════════════════════════════════════
# ── 逐层冻结 + 新层二维网格 (layer-wise grid) ─────────────────────────────
# 针对"数据饥饿"的正面解法。当前 BO 在 p=3 时要在 6 维里用 630 次评估找最优,
# 相当于每维只有 2.9 个采样点 —— 实测 BO 在 31% 预算处就平台了, 加预算无效。
#
# 这里改成: 冻结前 p-1 层已优化好的角度, **只搜新增那一层的 (γ_p, β_p)**。
#   · 搜索维度**永远是 2**, 跟 p 无关 —— 数据饥饿从根上消失
#   · n_grid × n_grid 的网格在二维里是穷举级覆盖(默认 40×40 = 1600 点)
#   · 网格包含 (γ_p, β_p) = (0,0), 而零时长的 cost/mixer 层就是恒等算符,
#     精确还原 p-1 层的电路 ⇒ **数学上保证 R_a(p) >= R_a(p-1)**, 顺带解决
#     我们表2b 记录的"高p反而更差"问题
#   · 网格是确定性的(BO 是随机的), 同一张图跑几次结果完全一样 ⇒ 不需要跑
#     多个 BO 种子取最好, n_bo_seeds=1 即可, 又省 5 倍
#   · 全程不用 GP, 省掉当前占 56% 运行时间的超参数拟合
# 这是 QAOA 数值工作的标准做法之一(与 Zhou et al. 的 INTERP 并列)。
#
# [三个 condition 在这里的区别] 网格搜索没有"初始化"这回事, 所以:
#   · cold_start / gnn_warmstart 的区别在**损失函数**(后者用有偏初态制备
#     + Egger 定制mixer), 这个区别在网格模式下完全保留, 是我们真正要比的东西
#   · tqa_init 的区别原本只在 BO 初始化 —— 网格穷举后这个区别消失了。这里
#     让它在精修阶段**额外**从 TQA 角度起一次, 所以 tqa_init >= cold_start
#     是构造上必然的。这个偏置对**基线有利**, 会让 gnn_warmstart 的优势显得
#     更保守, 结论因此更稳, 但报告时必须写明。
# ══════════════════════════════════════════════════════════════════════════
@everywhere function run_one_layerwise_chain(g, bo_seed::Int, p_max::Int;
                                              use_grid::Bool=false, n_grid::Int=40,
                                              layer_budget::Int=200, polish_iters::Int=300,
                                              n_polish_starts::Int=3, bo_verbose::Int=0)
    atoms, edges, weights = g.atoms, g.edges, g.weights
    n_atoms, total_w, denom = g.n_atoms, g.total_w, g.denom
    best_p_gnn = g.best_p_gnn

    CONDS = ["cold_start", "tqa_init", "gnn_warmstart"]
    out   = Dict(c => Dict{Int,Float64}() for c in CONDS)
    outw  = Dict(c => Dict{Int,Float64}() for c in CONDS)
    outt  = Dict(c => Dict{Int,Float64}() for c in CONDS)
    outx  = Dict(c => Dict{Int,Vector{Float64}}() for c in CONDS)   # 最优角度(查贴边用)
    prev  = Dict{String,Vector{Float64}}(c => Float64[] for c in CONDS)
    empty!(W_LOG_BUF)
    t_chain = time()

    for p_val in 1:p_max
        dim = 2*p_val
        lo  = zeros(dim)
        hi  = vcat([[BOX_HI[1], BOX_HI[2]] for _ in 1:p_val]...)
        delta_t = 0.75
        tqa = vcat([[delta_t*(k/p_val)*π, delta_t*(1-k/p_val)*π/2] for k in 1:p_val]...)

        for cnd in CONDS
            t_cond = time()
            tag = "[LW i$(g.inst)/s$(bo_seed)/p$(p_val)/$(cnd)]"
            Random.seed!(bo_seed*1000 + p_val*10 + findfirst(==(cnd), CONDS))

            cache = Dict{Vector{Float64},Float64}()
            function ev(x)
                xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
                haskey(cache, key) && return cache[key]
                v = try
                    lv = cnd == "gnn_warmstart" ?
                         loss_fn_warmstart_weighted(xs, best_p_gnn, atoms, edges, weights, n_atoms) :
                         loss_fn_weighted(xs, atoms, edges, weights, n_atoms)
                    isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
                catch; 1000.0; end
                cache[key] = v; return v
            end

            head = prev[cnd]                     # 冻结的前 p-1 层, 长度 2(p-1)
            lo2  = [0.0, 0.0]; hi2 = [BOX_HI[1], BOX_HI[2]]     # 新层的二维搜索域
            ev2(u) = ev(vcat(head, u))           # 嵌回全维度求值

            local by, bx, ar_stage1
            if use_grid
                # ── 变体 A: 纯二维网格(不用 GP, 最快, 但不体现 GSM+CSM) ──
                by = Inf; bx = zeros(dim)
                for γ in range(0.0, BOX_HI[1], length=n_grid), β in range(0.0, BOX_HI[2], length=n_grid)
                    x = vcat(head, [γ, β]); v = ev(x)
                    v < by && (by = v; bx = copy(x))
                end
                ar_stage1 = -by/denom
                bo_verbose > 0 && wlog(@sprintf("%s 网格 %d×%d: ar=%.4f", tag, n_grid, n_grid, ar_stage1), :detail)
            else
                # ── 变体 B(默认): 在新层的 2 维空间里跑 GSM+CSM 贝叶斯优化 ──
                # 这才是逐层分解的意义所在: 搜索维度恒为 2 ⇒ GP 超参数只有
                # (Q_g+Q_c)(1+2·2)=20 维, NelderMead 400 次迭代足以收敛(实测
                # p=1 时 5 个种子给出完全相同的结果), GSM+CSM 拟合多峰地形的
                # 能力才真正被用上。直接在 2p 维联合优化时, p=3 的超参数是
                # 52 维、只跑到收敛所需的 15%, 那个核实际上退化成了噪声。
                n_init2 = min(24, max(12, layer_budget ÷ 8))
                halton2 = halton_samples(n_init2, lo2, hi2; skip=(bo_seed*7)%100)
                init_us = Vector{Vector{Float64}}()
                # [关键] 显式把 (0,0) 放进初始设计。零时长的 cost/mixer 层就是
                # 恒等算符, 精确还原 p-1 层的电路 ⇒ 保证 R_a(p) >= R_a(p-1)。
                push!(init_us, [0.0, 0.0])
                # 上一层的 (γ,β) 作为提示点(layer-wise warm start)
                length(head) >= 2 && push!(init_us, clamp.(head[end-1:end], lo2, hi2))
                # [新增] 逐层贪心的风险: 某一层没找到最优, 会顺着冻结前缀锁死
                # 后面所有层(实测种子3、4 在 p=1 落后 0.014, 到 p=3 仍落后)。
                # 这里额外播撒几个覆盖 γ 全周期的点 —— QAOA 的 γ 方向往往有多个
                # 近似等价的盆地, 只从上一层附近出发容易困在其中一个。成本很低
                # (几个初始点), 但给了跳出坏盆地的机会。
                for gfrac in (0.25, 0.5, 0.75)
                    push!(init_us, [gfrac*3π, 0.5π])
                end
                for i in 1:(n_init2 - length(init_us))
                    push!(init_us, clamp.(halton2[i] .+ randn(2).*0.05.*(hi2.-lo2), lo2, hi2))
                end
                ys = Float64[ev2(u) for u in init_us]
                U = hcat(init_us...); y = copy(ys)
                by = minimum(ys); bu = init_us[argmin(ys)]

                Q_g=2; Q_c=2; D2=2
                xsc2 = hi2 .- lo2                               # [GP v2] 输入量程(v1 时被忽略)
                θ = map_spectral_hp(U,y,Q_g,Q_c,D2;nr=2,xscale=xsc2); stag=0
                for iter in 1:(layer_budget - n_init2)
                    t = iter/(layer_budget - n_init2)
                    # 热启动: 把上一次拟合的 θ 传进去当第一个起点(见 map_spectral_hp)
                    iter>1 && iter%18==0 && (θ = map_spectral_hp(U,y,Q_g,Q_c,D2;nr=2,θ_init=θ,xscale=xsc2))
                    gp = build_spectral_gp(U,y,θ,Q_g,Q_c,D2; xscale=xsc2)
                    C  = lo2 .+ rand(D2,400).*(hi2.-lo2)
                    un = if t < 0.30
                        C[:,ts_acq(gp,C;ns=10)]
                    elseif t < 0.68
                        ξ = 0.05*exp(-3.0*(t-0.30)/0.38)
                        μ,σ = gp_pred(gp,C); C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
                    else
                        β_l = max(exp(-2.5*(t-0.68)/0.32),0.05)
                        μ,σ = gp_pred(gp,C); C[:,argmin(lcb_acq(μ,σ;β=β_l))]
                    end
                    yn = ev2(un); U = hcat(U,un); push!(y,yn)
                    if yn < by; by = yn; bu = copy(un); stag = 0
                    else; t >= 0.30 && (stag += 1); end
                    if stag >= 20
                        for _ in 1:2
                            ur = lo2 .+ rand(D2).*(hi2.-lo2); yr = ev2(ur)
                            U = hcat(U,ur); push!(y,yr); yr < by && (by = yr; bu = copy(ur))
                        end; stag = 0
                    end
                    # [FIX] 循环内多尺度 NelderMead 精修。联合BO 一直有这一步
                    # (每15次迭代一轮), 我第一版逐层实现把精修全挪到最后了 ——
                    # 后果: p=1 面对的是**完全相同**的二维问题, 联合BO 5/5 个种子
                    # 都命中最优, 逐层却只有 3/5(另外2个卡在 0.8434 而不是 0.8570),
                    # 而且这个失手会顺着冻结前缀一路传到 p=2、p=3。
                    # BO 的采集函数负责全局探索, 局部精修负责把候选点推到真正的
                    # 极值点上, 两者缺一不可 —— 只在最后精修一次, 中途那些"差一点
                    # 就能到更好盆地"的候选点就被浪费了。
                    if iter % 15 == 0 && t >= 0.30
                        for sc in (0.05, 0.20, 0.50)
                            u0 = clamp.(bu .+ randn(D2).*sc.*(hi2.-lo2), lo2.+1e-6, hi2.-1e-6)
                            rr = optimize(u -> ev2(clamp.(u, lo2, hi2)), u0, NelderMead(),
                                          Optim.Options(iterations=100, f_reltol=1e-7))
                            yr = Optim.minimum(rr); ur = clamp.(Optim.minimizer(rr), lo2, hi2)
                            if yr < by - 1e-6
                                by = yr; bu = ur; stag = 0
                                U = hcat(U,ur); push!(y,yr)
                            end
                        end
                    end
                end
                bx = vcat(head, bu)
                ar_stage1 = -by/denom
                bo_verbose > 0 && wlog(@sprintf("%s 二维GSM+CSM BO (budget=%d): ar=%.4f",
                                                tag, layer_budget, ar_stage1), :detail)
            end

            # ── 全维度联合精修(只接受改进 ⇒ 不破坏单调性) ──────────────
            # [FIX] 这一步是逐层策略里**唯一能挣脱冻结前缀**的环节, 原来的扰动
            # 幅度只有 0.03/0.06 倍量程, 太局部, 跳不出前缀锁定的盆地。
            # 实测证据: 实例3 的 p=3, 逐层的 5 个种子**全部一致**给出 0.8914,
            # 而联合BO 找到 0.9029 —— 种子一致 ≠ 找到全局最优, 它们是被同一个
            # 前缀一起锁在了次优点上。差距随 p 单调增大(p=1 完全一致, p=3 出现
            # -0.0079 / -0.0115), 正是前缀约束的特征。
            # 改成阶梯式扰动: 从局部微调一直到半个量程的大跳, 让精修既能收敛
            # 又有机会重新安排**前面几层**的角度。只接受改进, 单调性不受影响。
            POLISH_SCALES = (0.05, 0.15, 0.30, 0.50)
            starts = Vector{Vector{Float64}}([copy(bx)])
            for (k, sc) in enumerate(POLISH_SCALES)
                for rep in 1:max(1, n_polish_starts ÷ 2)
                    pert = [((i*7 + k*13 + rep*37 + p_val*29) % 100)/100 - 0.5 for i in 1:dim]
                    push!(starts, clamp.(bx .+ pert .* sc .* (hi.-lo), lo, hi))
                end
            end
            # [新增] INTERP 起点(Zhou et al.): 把 p-1 层的整条角度序列**线性插值**
            # 成 p 层, 而不是"保留前 p-1 层原样 + 末尾加一层"。两者的区别正是
            # 前缀约束: 逐层冻结要求前 p-1 层一字不变, INTERP 允许整条 schedule
            # 平滑地重新分布。作为额外的精修起点加进来, 成本一个起点, 但给了
            # 完全不同的一种"挣脱前缀"的方式。
            if p_val >= 2 && length(head) == 2*(p_val-1)
                γ_old = head[1:2:end]; β_old = head[2:2:end]
                interp(v, m) = [begin
                    t = (i-1)/(m-1) * (length(v)-1) + 1
                    lo_i = clamp(floor(Int,t), 1, length(v)); hi_i = clamp(lo_i+1, 1, length(v))
                    f = t - lo_i
                    (1-f)*v[lo_i] + f*v[hi_i]
                end for i in 1:m]
                γ_new = interp(γ_old, p_val); β_new = interp(β_old, p_val)
                push!(starts, clamp.(vcat([[γ_new[i], β_new[i]] for i in 1:p_val]...), lo, hi))
            end
            # tqa_init 额外从 TQA 角度起一次 —— 这个偏置对**基线有利**, 会让
            # gnn_warmstart 的优势显得更保守, 报告时须写明。
            cnd == "tqa_init" && push!(starts, clamp.(tqa, lo, hi))

            for x0 in starts
                r = optimize(x -> ev(clamp.(x, lo, hi)), x0, NelderMead(),
                             Optim.Options(iterations=polish_iters, f_reltol=1e-8))
                yv = Optim.minimum(r)
                yv < by - 1e-9 && (by = yv; bx = clamp.(Optim.minimizer(r), lo, hi))
            end

            out[cnd][p_val]  = -by/denom
            outw[cnd][p_val] = -by/total_w
            outt[cnd][p_val] = time() - t_cond
            prev[cnd] = copy(bx)
            outx[cnd][p_val] = copy(bx)

            wlog(@sprintf("      · inst%d/p=%d %-14s ar=%.4f (/Σw=%.4f)  [搜索%.4f→精修%.4f]  [%.1f分钟]",
                          g.inst, p_val, cnd, out[cnd][p_val], outw[cnd][p_val],
                          ar_stage1, out[cnd][p_val], outt[cnd][p_val]/60), :summary)
        end
    end
    return (inst=g.inst, bo_seed=bo_seed, out=out, outw=outw, outt=outt,
            logbuf=copy(W_LOG_BUF), outx=outx)
end


# ══════════════════════════════════════════════════════════════════════════
# ── 一个 (实例, BO种子) 任务: 独立跑完 p=1..p_max × 3个condition ──────────
# 这是为**集群**重写的并行粒度。原来的写法只在"BO种子"这一层 pmap, 意味着
# n_bo_seeds=5 时不管集群给多少核, 也只有 5 个在干活。现在把
# (实例 × BO种子) 摊平, 任务数 = n_instances × n_bo_seeds (10×5=50), 能喂饱
# 几十个核。
# [跟原来的一处行为差别] 原来 p 层之间的"上一层最优角度"提示(prev_best)取自
# **所有种子里最好的那个**, 现在每个种子链自己的 —— 因为种子之间已经并行、
# 互相看不到对方。这其实是更干净的设计(每个种子 = 一次完全独立的重启), 但
# 数字会跟之前串行版略有不同, 写论文时按现在这版重跑一遍即可。
# ══════════════════════════════════════════════════════════════════════════
@everywhere function run_one_bo_chain(g, bo_seed::Int, p_max::Int; budget_scale::Float64=1.0,
                                      bo_verbose::Int=0)
    atoms, edges, weights = g.atoms, g.edges, g.weights
    n_atoms, total_w, denom = g.n_atoms, g.total_w, g.denom
    best_p_gnn = g.best_p_gnn

    CONDS = ["cold_start", "tqa_init", "gnn_warmstart"]
    empty!(W_LOG_BUF)      # 每个任务独立的日志缓冲, 结束时随返回值带回主进程
    out  = Dict(c => Dict{Int,Float64}() for c in CONDS)   # /C_max
    outw = Dict(c => Dict{Int,Float64}() for c in CONDS)   # /Σw
    outt = Dict(c => Dict{Int,Float64}() for c in CONDS)   # 该 (p,cond) 的CPU秒数
    prev = Dict{String,Any}(c => nothing for c in CONDS)
    t_chain = time()

    for p_val in 1:p_max
        dim    = 2*p_val
        lo     = zeros(dim)
        hi     = vcat([[BOX_HI[1], BOX_HI[2]] for _ in 1:p_val]...)
        # budget_scale<1 用于本地冒烟测试(几分钟跑完整条流程验证代码), 正式
        # 出数据必须用 1.0 —— 缩小的 budget 下 BO 根本没收敛, 数字没有意义。
        budget = max(30, round(Int, budget_scale * (BO_BUDGET_A*dim + BO_BUDGET_B*dim)))
        n_init = max(10, round(Int, budget_scale * 8*dim))
        delta_t = 0.75
        tqa_angles = vcat([[delta_t*(k/p_val)*π, delta_t*(1-k/p_val)*π/2] for k in 1:p_val]...)

        for cnd in CONDS
            t_cond = time()
            Random.seed!(bo_seed*1000 + p_val*10 + findfirst(==(cnd), CONDS))
            cache = Dict{Vector{Float64},Float64}()
            function ev!(x)
                xs = clamp.(x, lo, hi); key = round.(xs, digits=6)
                haskey(cache, key) && return cache[key]
                try
                    lv = if cnd == "gnn_warmstart"
                        loss_fn_warmstart_weighted(xs, best_p_gnn, atoms, edges, weights, n_atoms)
                    else
                        loss_fn_weighted(xs, atoms, edges, weights, n_atoms)
                    end
                    v = isfinite(lv) ? clamp(lv, -(total_w+1.0), 1000.0) : 1000.0
                    cache[key] = v; return v
                catch; return 1000.0; end
            end

            # [FIX] 跨 p 层传递上一层最优角度。原来的判据是
            #   !isnothing(hint) && length(hint) == dim
            # 但 prev[cnd] 长度是 2(p-1), dim 是 2p —— **永远对不上**, 于是每一层
            # 都退化成从零开始的纯 Halton 初始化。后果: p 越大维度越高, BO 裸搜
            # 越搜不动, 实测出现 p=4 的 ar 反而低于 p=1(0.8551 < 0.8570)。
            # 现在按 QAOA 的层结构把上一层角度补齐到新维度(Zhou et al. 的
            # INTERP 思路): 角度向量是 [γ1,β1,γ2,β2,...], 在末尾复制最后一层的
            # (γ,β) 作为新增层的初值 —— 新层从"跟上一层相同"开始, 是这类
            # layer-wise warm start 里最稳的一种。
            hint = if cnd == "tqa_init"
                tqa_angles
            elseif !isnothing(prev[cnd]) && length(prev[cnd]) == dim - 2
                vcat(prev[cnd], prev[cnd][end-1:end])   # 复制最后一层的 (γ,β)
            elseif !isnothing(prev[cnd]) && length(prev[cnd]) == dim
                prev[cnd]
            else
                nothing
            end
            halton_pts = halton_samples(n_init, lo, hi; skip=(bo_seed*7)%100)
            init_xs = Vector{Vector{Float64}}(undef, n_init)
            if !isnothing(hint) && length(hint) == dim
                n_h = max(3, n_init÷3)
                for i in 1:n_h
                    sc = 0.05 + 0.10*(i-1)/n_h
                    init_xs[i] = clamp.(hint .+ randn(dim).*sc.*(hi.-lo), lo, hi)
                end
                for i in (n_h+1):n_init
                    init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                end
            else
                for i in 1:n_init
                    init_xs[i] = clamp.(halton_pts[i].+randn(dim).*0.05.*(hi.-lo), lo, hi)
                end
            end
            init_ys = Float64[ev!(x) for x in init_xs]
            X = hcat(init_xs...); y = copy(init_ys)
            by = minimum(init_ys); bx = init_xs[argmin(init_ys)]
            # BO 内部过程打印(默认关闭)。bo_verbose=N 表示每 N 次迭代打印一行,
            # 用来观察: 初始设计给到什么水平、三个采集阶段(Thompson采样→EI→LCB)
            # 各自贡献多少提升、停滞重启和 NelderMead 精修有没有起作用。
            tag_bo = "[BO i$(g.inst)/s$(bo_seed)/p$(p_val)/$(cnd)]"
            if bo_verbose > 0
                wlog(@sprintf("%s 初始设计 %d 点(%s): best ar=%.4f", tag_bo, n_init,
                        isnothing(hint) ? "纯Halton" : "含上一层角度提示", -by/denom), :detail)
            end
            n_imp = 0; n_restart = 0; n_refine_hit = 0
            # 心跳默认 300 秒(5分钟)。原来是 120 秒, 但按 20实例×5种子×3条件估算,
            # 心跳会产生约 3600 行 ≈ 320KB, 加上其它输出就逼近 Jupyter 的 512KB
            # 上限了。5 分钟一次既够判断"还在跑", 又把控制台用量压到安全区间。
            hb_sec = something(tryparse(Float64, get(ENV,"JULIA_HEARTBEAT_SEC","")), 300.0)
            t_hb = time()

            Q_g=2; Q_c=2
            # [FIX] 难度档位的判据原来是 dim>=8, 而 dim=2p, 所以 p=3(dim=6)
            # 落在低维档 —— 但 p=3 恰恰是我们实际跑的最难的情况, 却在用跟
            # p=1 一样的配置(候选池400、NelderMead只迭代100次、精修尺度只有
            # 3档粗网格)。这个阈值应该是当初为 Q3 那套实验定的, 后来 p 的范围
            # 变了没跟着改。改成 dim>=BO_DIM_THRESH 让 p=3 吃到高维档配置。
            # [注意] 这改变了 p=3 的数值结果, 已有的 p=3 checkpoint 会与新
            # 配置不一致 —— 想要一致的数据集, 删掉 ckpt 目录重跑。
            n_cand       = dim>=BO_DIM_THRESH ? 800 : 400
            map_every    = dim>=BO_DIM_THRESH ? 12 : 18
            phase1_end   = dim>=BO_DIM_THRESH ? 0.45 : 0.30
            phase2_end   = dim>=BO_DIM_THRESH ? 0.78 : 0.68
            nr_hp        = dim>=BO_DIM_THRESH ? 4 : 2
            stag_limit   = dim>=BO_DIM_THRESH ? 25 : 20
            refine_every = 15
            nm_iters     = dim>=BO_DIM_THRESH ? 200 : 100
            refine_scales = dim>=BO_DIM_THRESH ? [0.02,0.05,0.15,0.40,1.0] : [0.05,0.20,0.50]
            xscd = hi .- lo                                 # [GP v2] 输入量程(v1 时被忽略)
            θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp,xscale=xscd); stag=0

            for iter in 1:(budget-n_init)
                t = iter/(budget-n_init)
                iter>1 && iter%map_every==0 && (θ_gp = map_spectral_hp(X,y,Q_g,Q_c,dim;nr=nr_hp,xscale=xscd))
                gp = build_spectral_gp(X,y,θ_gp,Q_g,Q_c,dim; xscale=xscd)
                C = lo.+rand(dim,n_cand).*(hi.-lo)
                if t < phase1_end
                    xn = C[:,ts_acq(gp,C;ns=10)]
                elseif t < phase2_end
                    ξ = 0.05*exp(-3.0*(t-phase1_end)/(phase2_end-phase1_end))
                    μ,σ = gp_pred(gp,C); xn = C[:,argmax(ei_acq(μ,σ,by;ξ=ξ))]
                else
                    β = max(exp(-2.5*(t-phase2_end)/(1-phase2_end)),0.05)
                    μ,σ = gp_pred(gp,C); xn = C[:,argmin(lcb_acq(μ,σ;β=β))]
                end
                yn = ev!(xn); X = hcat(X,xn); push!(y,yn)
                improved = yn < by
                if improved; by = yn; bx = copy(xn); stag = 0; n_imp += 1
                else; t >= phase1_end && (stag += 1); end
                if stag >= stag_limit
                    for _ in 1:2
                        xr = lo.+rand(dim).*(hi.-lo); yr = ev!(xr)
                        X = hcat(X,xr); push!(y,yr); yr < by && (by = yr; bx = copy(xr))
                    end; stag = 0; n_restart += 1
                    bo_verbose > 0 && wlog(@sprintf("%s   └ 停滞%d次, 随机重启", tag_bo, stag_limit), :detail)
                end
                # ── 心跳 ──────────────────────────────────────────────────
                # 不受 bo_verbose 控制, 默认每 120 秒必打一行。原因: 进度打印
                # 原本只在"一个(p,condition)跑完"时触发, 而这个单元的耗时随 p
                # 迅速变长(p=1约4分钟, p=2约20分钟, p=3约60分钟) —— 中间那段
                # 长时间静默跟卡死无法区分, 实测已经导致多次误判中断。
                # 间隔可用 JULIA_HEARTBEAT_SEC 调整, 设 0 关闭。
                if hb_sec > 0 && time() - t_hb >= hb_sec
                    wlog(@sprintf("%s ...运行中 iter %d/%d (%.0f%%) best=%.4f 已用%.1f分钟",
                            tag_bo, iter, budget-n_init, 100*iter/(budget-n_init),
                            -by/denom, (time()-t_cond)/60), :summary)
                    t_hb = time()
                end
                if bo_verbose > 0 && (iter % bo_verbose == 0 || improved)
                    phase = t < phase1_end ? "Thompson" : (t < phase2_end ? "EI      " : "LCB     ")
                    wlog(@sprintf("%s iter %4d/%4d %s 本次ar=%.4f best=%.4f%s stag=%2d 观测%d点",
                            tag_bo, iter, budget-n_init, phase, -yn/denom, -by/denom,
                            improved ? " ↑" : "  ", stag, length(y)), :detail)
                end
                if iter%refine_every==0 && t >= phase1_end
                    for sc in refine_scales
                        xs0 = clamp.(bx.+randn(dim).*sc.*(hi.-lo), lo.+1e-6, hi.-1e-6)
                        r = optimize(x->ev!(clamp.(x,lo,hi)), xs0, NelderMead(),
                                     Optim.Options(iterations=nm_iters,f_reltol=1e-7,show_trace=false))
                        yr = Optim.minimum(r); xr = clamp.(Optim.minimizer(r),lo,hi)
                        if yr < by-1e-6
                            by = yr; bx = xr; stag = 0; X = hcat(X,xr); push!(y,yr)
                            n_refine_hit += 1
                            bo_verbose > 0 && wlog(@sprintf("%s   └ NelderMead精修(scale=%.2f) → %.4f",
                                                       tag_bo, sc, -by/denom), :detail)
                        end
                    end
                end
            end
            if bo_verbose > 0
                wlog(@sprintf("%s 结束: best ar=%.4f | 采集提升%d次, 停滞重启%d次, 精修命中%d次, 总观测%d点",
                        tag_bo, -by/denom, n_imp, n_restart, n_refine_hit, length(y)), :detail)
            end
            out[cnd][p_val]  = -by/denom
            outw[cnd][p_val] = -by/total_w
            outt[cnd][p_val] = time() - t_cond
            prev[cnd] = bx
            # 实时进度: 摊平并行之后任务是乱序完成的, 这行只是为了"看得出还活着"
            # (单个任务要跑好几个小时, 没有这行跟卡死无法区分)。真正给人看的、
            # 按 实例→p→condition 排好序的完整输出, 在 pmap 全部结束之后统一打印。
            wlog(@sprintf("      · inst%d/boseed%d  p=%d %-14s ar=%.4f (/Σw=%.4f)  [本段%.1f分钟]",
                    g.inst, bo_seed, p_val, cnd, out[cnd][p_val], outw[cnd][p_val],
                    outt[cnd][p_val]/60), :summary)
        end
    end
    return (inst=g.inst, bo_seed=bo_seed, out=out, outw=outw, outt=outt,
            logbuf=copy(W_LOG_BUF))
end
"""从 (GRAPHS, RES) 生成全套论文报表 + CSV。抽成函数是为了让"跑完直接出表"
   和"从 checkpoint 事后出表"两条路径共用同一份逻辑, 不会分叉。
   tag 会拼进输出文件名, 用来区分不同配置的结果(比如新旧 BO 阈值), 避免覆盖。"""
function emit_paper_report(GRAPHS, RES, p_max::Int, n_at::Int, n_bo_seeds::Int,
                            optimizer::String, tag::String="")
    # ── 聚合: 每个实例内部对 BO 种子取 best ─────────────────────────────
    CONDS = ["cold_start", "tqa_init", "gnn_warmstart"]
    sfx = isempty(tag) ? "" : "_" * tag      # 拼进输出文件名, 区分不同配置
    inst_best  = Dict(c => Dict(p => Float64[] for p in 1:p_max) for c in CONDS)
    inst_bestw = Dict(c => Dict(p => Float64[] for p in 1:p_max) for c in CONDS)
    # [新增] 累积最优 max_{p'<=p} R_a(p')。
    # 依据: p 层 QAOA 的电路能精确退化成 p-1 层(把第 p 层的 γ,β 置零, 两个
    # 演化算符变成恒等), 所以 p-1 层的最优解是 p 层搜索空间里的一个合法点,
    # 真实最优必然满足 R_a(p) >= R_a(p-1) —— 单调不降是**物理约束**。
    # 但实测 BO 在高维搜不动(p=3 是6维, budget 只有582, 每维平均不到3个采样
    # 点), 会报出低于 p-1 的结果, 那纯粹是数值优化失败, 不是物理效应。
    # [重要] 这一列**不覆盖**原始值。主表照实报 BO 搜到的 inst_best, 这一列
    # 单独列出, 两者的差就是"经典优化器在该 p 上失效了多少" —— 这个差本身
    # 是应该报告的信息, 覆盖掉反而掩盖了真相。写论文时两列都给, 并说明来源。
    inst_cummax = Dict(c => Dict(p => Float64[] for p in 1:p_max) for c in CONDS)
    for g in GRAPHS
        rs = [r for r in RES if r.inst == g.inst]
        for c in CONDS, p in 1:p_max
            push!(inst_best[c][p],  maximum(r.out[c][p]  for r in rs))
            push!(inst_bestw[c][p], maximum(r.outw[c][p] for r in rs))
        end
        for c in CONDS
            run_max = -Inf
            for p in 1:p_max
                run_max = max(run_max, inst_best[c][p][end])
                push!(inst_cummax[c][p], run_max)
            end
        end
    end
    n_done = length(GRAPHS)

    # ══════════════════════════════════════════════════════════════════
    # 【详细日志】把乱序完成的并行结果, 按 实例 → p → condition → 逐task
    # 重新排好序完整打印一遍 —— 格式跟串行版一致, 便于逐条核对。
    # 并行是为了喂饱集群, 但人读的输出不该跟着乱序。
    # ══════════════════════════════════════════════════════════════════
    mprintln("\n\n" * "="^78)
    mprintln("【详细日志】逐实例 / 逐p / 逐condition / 逐BO种子")
    mprintln("="^78)
    for g in GRAPHS
        rs = sort([r for r in RES if r.inst == g.inst], by = r -> r.bo_seed)
        @printf("\n[实例 %d/%d] seed=%d  |E|=%d  C_max=%.4f  C_max/Σw=%.4f  unwanted_ratio=%.4f  权重相关系数=%.4f  GNN(/Σw)=%.4f  GNN(/C_max)=%.4f\n",
                g.inst, n_done, g.seed, g.n_edges, g.cmax, g.cmax/g.total_w,
                g.unwanted, g.fidelity, g.gnn_ar_w, g.gnn_ar_cmax)
        for p in 1:p_max
            for c in CONDS
                @printf("\n  ── p=%d %s (共 %d 个BO种子) ──\n", p, c, n_bo_seeds)
                for r in rs
                    @printf("    task=%d [%s] p=%d ar=%.4f  (/Σw=%.4f)  [%.1f分钟]\n",
                            r.bo_seed, c, p, r.out[c][p], r.outw[c][p], r.outt[c][p]/60)
                end
                vals  = [r.out[c][p]  for r in rs]
                valsw = [r.outw[c][p] for r in rs]
                tsum  = sum(r.outt[c][p] for r in rs)/60
                @printf("    [%s] p=%d: best=%.4f  mean=%.4f±%.4f  (/Σw best=%.4f)  [CPU合计%.1f分钟]\n",
                        c, p, maximum(vals), mean(vals),
                        (length(vals)>1 ? std(vals) : 0.0), maximum(valsw), tsum)
            end
        end
        @printf("\n  [实例 %d 小结] ", g.inst)
        for p in 1:p_max
            @printf("p%d: cold=%.4f/tqa=%.4f/gnn=%.4f  ", p,
                    inst_best["cold_start"][p][g.inst], inst_best["tqa_init"][p][g.inst],
                    inst_best["gnn_warmstart"][p][g.inst])
        end
        mprintln()
    end

    # ══════════════════════════════════════════════════════════════════
    # 论文用数据: 全部打印出来
    # ══════════════════════════════════════════════════════════════════
    mprintln("\n\n" * "="^78)
    mprintln("【表1】实例属性 (论文里描述测试集用)")
    mprintln("="^78)
    mprintln(" 实例 seed |E| |  C_max   Σw    C_max/Σw | unwanted 权重相关 权重范围      | GNN/C_max")
    mprintln("-"^78)
    for g in GRAPHS
        @printf(" %3d %4d %3d | %7.4f %7.4f %.4f | %.4f  %.4f  [%.3f,%.3f] | %.4f\n",
                g.inst, g.seed, g.n_edges, g.cmax, g.total_w, g.cmax/g.total_w,
                g.unwanted, g.fidelity, g.wmin, g.wmax, g.gnn_ar_cmax)
    end
    mprintln("-"^78)
    @printf(" 均值±标准差: C_max/Σw=%.4f±%.4f  unwanted=%.4f±%.4f  权重相关=%.4f±%.4f\n",
            mean(g.cmax/g.total_w for g in GRAPHS), std([g.cmax/g.total_w for g in GRAPHS]),
            mean(g.unwanted for g in GRAPHS), std([g.unwanted for g in GRAPHS]),
            mean(g.fidelity for g in GRAPHS), std([g.fidelity for g in GRAPHS]))
    @printf(" 纯经典 LGNN (/C_max): %.4f±%.4f   (=1.0 表示该实例上找到了精确最优)\n",
            mean(g.gnn_ar_cmax for g in GRAPHS), std([g.gnn_ar_cmax for g in GRAPHS]))
    @printf(" 其中 %d/%d 个实例 LGNN 直接找到了精确最优解\n",
            count(g -> g.gnn_ar_cmax > 0.9999, GRAPHS), n_done)

    mprintln("\n" * "="^78)
    mprintln("【表2】主结果: 近似比 R_a = <C>/C_max  (Graham et al. arXiv:2112.14589 定义)")
    mprintln("每格 = 跨 $(n_done) 个实例的 mean ± std (每实例内部已对 $(n_bo_seeds) 个BO种子取best)")
    mprintln("="^78)
    mprintln("  p │  cold_start     │   tqa_init      │  gnn_warmstart  │ 配对优势(gnn-max(cold,tqa))")
    mprintln("-"^78)
    for p in 1:p_max
        mc,sc = mean(inst_best["cold_start"][p]),    (n_done>1 ? std(inst_best["cold_start"][p]) : 0.0)
        mt,st = mean(inst_best["tqa_init"][p]),      (n_done>1 ? std(inst_best["tqa_init"][p]) : 0.0)
        mg,sg = mean(inst_best["gnn_warmstart"][p]), (n_done>1 ? std(inst_best["gnn_warmstart"][p]) : 0.0)
        diffs = [inst_best["gnn_warmstart"][p][k] - max(inst_best["cold_start"][p][k],
                                                        inst_best["tqa_init"][p][k]) for k in 1:n_done]
        @printf("  %d │ %.4f ± %.4f │ %.4f ± %.4f │ %.4f ± %.4f │ %+.4f ± %.4f  (%d/%d 赢)\n",
                p, mc,sc, mt,st, mg,sg, mean(diffs),
                (n_done>1 ? std(diffs) : 0.0), count(>(0.0), diffs), n_done)
    end

    # ── 表2b: 累积最优 + 数值优化失效量 ─────────────────────────────────
    mprintln("\n【表2b】累积最优 max over p'<=p of R_a(p')")
    mprintln("上表是 BO 在该 p 实际搜到的原始值; 这张表是\"截至 p 层的最好成绩\"。")
    mprintln("物理上 R_a(p) 必然 >= R_a(p-1) —— p 层电路把第 p 层角度置零就精确")
    mprintln("退化成 p-1 层, 所以 p-1 的最优解是 p 层搜索空间里的合法点。两表之差")
    mprintln("就是经典优化器在该 p 上失效的幅度。论文里两个数字都应该报, 后者不能")
    mprintln("替代前者(替代会掩盖\"高p优化变难\"这个真实且值得报告的信息)。")
    mprintln("-"^78)
    mprintln("  p │  cold_start     │   tqa_init      │  gnn_warmstart  │ gnn失效量(累积-原始)")
    mprintln("-"^78)
    for p in 1:p_max
        cc = mean(inst_cummax["cold_start"][p]);    sc2 = n_done>1 ? std(inst_cummax["cold_start"][p]) : 0.0
        ct = mean(inst_cummax["tqa_init"][p]);      st2 = n_done>1 ? std(inst_cummax["tqa_init"][p]) : 0.0
        cg = mean(inst_cummax["gnn_warmstart"][p]); sg2 = n_done>1 ? std(inst_cummax["gnn_warmstart"][p]) : 0.0
        # 逐实例失效量: 该实例上"累积最优"高出"本层原始值"多少。
        # >0 表示 BO 在这一层搜出来的还不如更低的层, 即这一层优化失败了。
        gap = [inst_cummax["gnn_warmstart"][p][k] - inst_best["gnn_warmstart"][p][k] for k in 1:n_done]
        n_fail = count(x -> x > 1e-9, gap)
        @printf("  %d │ %.4f ± %.4f │ %.4f ± %.4f │ %.4f ± %.4f │ %+.4f  (%d/%d 实例该层没超过更低层)\n",
                p, cc,sc2, ct,st2, cg,sg2, mean(gap), n_fail, n_done)
    end

    # ── 表2c: 最优角度是否贴在参数盒上界 ─────────────────────────────────
    # 参数空间不是周期的(见 QAOA_BOX_SCALE 说明), [0,3π]×[0,π] 只是搜索盒。
    # 时长下界 0 是物理边界(不能为负), 贴下界不算问题; 贴**上界**则说明最优点
    # 可能在盒外, 放大盒子(QAOA_BOX_SCALE=2)有可能更好。旧版结果没存角度, 跳过。
    if any(r -> hasproperty(r, :outx), RES)
        mprintln("\n【表2c】最优角度贴在参数盒**上界**的实例数 (每层盒 = [0,$(round(BOX_HI[1],digits=3))]×[0,$(round(BOX_HI[2],digits=3))], 容差 1e-3)")
        mprintln("  时长下界 0 是物理边界, 不计; 贴上界说明更大的盒子可能更好(可用 QAOA_BOX_SCALE=2 检验)")
        for p in 1:p_max
            hiv = vcat([[BOX_HI[1], BOX_HI[2]] for _ in 1:p]...)
            parts = String[]
            for c in CONDS
                n_hit = 0; n_tot = 0
                for g in GRAPHS
                    rs = [r for r in RES if r.inst == g.inst && hasproperty(r, :outx) && haskey(r.outx[c], p)]
                    isempty(rs) && continue
                    rb = rs[argmax([r.out[c][p] for r in rs])]     # 该实例取 best 的那个种子
                    n_tot += 1
                    any(rb.outx[c][p] .>= hiv .- 1e-3) && (n_hit += 1)
                end
                push!(parts, "$(c)=$(n_hit)/$(n_tot)")
            end
            mprintln("  p=$(p): " * join(parts, "   "))
        end
    end

    mprintln("\n" * "="^78)
    mprintln("【表3】逐实例明细 (论文附录 / 画箱线图用)")
    mprintln("="^78)
    for p in 1:p_max
        mprintln(" p=$(p):")
        mprintln("   实例 | cold_start  tqa_init  gnn_warmstart | 优势")
        for k in 1:n_done
            cd = inst_best["cold_start"][p][k]; tq = inst_best["tqa_init"][p][k]
            gw = inst_best["gnn_warmstart"][p][k]
            @printf("   %3d  |  %.4f     %.4f     %.4f      | %+.4f\n",
                    GRAPHS[k].seed, cd, tq, gw, gw - max(cd,tq))
        end
    end

    mprintln("\n" * "="^78)
    mprintln("【表4】关键论点的数据支撑")
    mprintln("="^78)
    # 论点A: warmstart 的低 p 能不能打平/打败 baseline 的高 p (= 省电路深度)
    mprintln(" (A) 电路深度节省: warmstart 的低 p 能不能顶 cold_start 的高 p")
    let cold_best_p = argmax([mean(inst_best["cold_start"][j]) for j in 1:p_max]),
        cold_best_v = maximum(mean(inst_best["cold_start"][j]) for j in 1:p_max)
        @printf("     cold_start 在测试的 p=1..%d 中最好的是 p=%d, R_a=%.4f\n",
                p_max, cold_best_p, cold_best_v)
        for i in 1:p_max
            gi = mean(inst_best["gnn_warmstart"][i])
            if gi > cold_best_v
                @printf("     → warmstart p=%d (%.4f) 已经超过 cold_start 的最好成绩 (+%.4f)\n",
                        i, gi, gi - cold_best_v)
            else
                @printf("       warmstart p=%d (%.4f) 还没超过 cold_start 最好成绩\n", i, gi)
            end
        end
        # 逐实例配对版本: 更严格, 回答"每一张图上都成立吗"
        for i in 1:p_max
            n_beat = count(k -> inst_best["gnn_warmstart"][i][k] >
                                maximum(inst_best["cold_start"][j][k] for j in 1:p_max), 1:n_done)
            @printf("       [逐实例配对] warmstart p=%d 打败该图上 cold_start 全部 p 的最好值: %d/%d\n",
                    i, n_beat, n_done)
        end
    end
    # 论点B: tqa 相对 cold 有没有增益(反衬 warmstart 的增益是真实的)
    mprintln(" (B) TQA 初始化相对 cold_start 的增益(用来反衬 warmstart 的增益不是'随便换个初始化都行'):")
    for p in 1:p_max
        d = [inst_best["tqa_init"][p][k] - inst_best["cold_start"][p][k] for k in 1:n_done]
        # [FIX] 原来用 >0 判定"更好", 但两者常常只差浮点尾数(实测出现过
        # "+0.0000±0.0000 却说 2/2 更好"的自相矛盾输出)。改成带容差, 并单独
        # 报"持平"的数量 —— 持平本身就是这里最想说明的结论。
        tol = 1e-6
        @printf("     p=%d: %+.6f ± %.6f  (更好 %d / 持平 %d / 更差 %d, 共 %d 个实例)\n",
                p, mean(d), (n_done>1 ? std(d) : 0.0),
                count(x -> x >  tol, d), count(x -> abs(x) <= tol, d),
                count(x -> x < -tol, d), n_done)
    end
    # 论点C: 随 p 的增长率
    mprintln(" (C) 每增加一层带来的增益:")
    for p in 2:p_max
        dc = mean(inst_best["cold_start"][p]) - mean(inst_best["cold_start"][p-1])
        dg = mean(inst_best["gnn_warmstart"][p]) - mean(inst_best["gnn_warmstart"][p-1])
        @printf("     p=%d→%d: cold_start %+.4f, gnn_warmstart %+.4f\n", p-1, p, dc, dg)
    end
    mprintln("="^78)

    # ── CSV 导出(画图用) ────────────────────────────────────────────────
    mkpath(directoryBasis)
    csv1 = joinpath(directoryBasis, "paper_results_n$(n_at)_inst$(n_done)$(sfx).csv")
    open(csv1, "w") do io
        println(io, "instance,seed,n_atoms,n_edges,cmax,total_w,cmax_over_w,unwanted_ratio," *
                    "weight_fidelity,gnn_ar_cmax,gnn_ar_sigmaw,p,condition,ar_cmax,ar_sigmaw,ar_cmax_cummax")
        for (k,g) in enumerate(GRAPHS), p in 1:p_max, c in CONDS
            @printf(io, "%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%s,%.6f,%.6f,%.6f\n",
                    g.inst, g.seed, g.n_atoms, g.n_edges, g.cmax, g.total_w, g.cmax/g.total_w,
                    g.unwanted, g.fidelity, g.gnn_ar_cmax, g.gnn_ar_w,
                    p, c, inst_best[c][p][k], inst_bestw[c][p][k], inst_cummax[c][p][k])
        end
    end
    csv2 = joinpath(directoryBasis, "paper_raw_n$(n_at)_inst$(n_done)$(sfx).csv")
    open(csv2, "w") do io
        println(io, "instance,seed,bo_seed,p,condition,ar_cmax,ar_sigmaw,cpu_seconds")
        for r in RES
            g = GRAPHS[r.inst]
            for p in 1:p_max, c in CONDS
                @printf(io, "%d,%d,%d,%d,%s,%.6f,%.6f,%.1f\n", r.inst, g.seed, r.bo_seed,
                        p, c, r.out[c][p], r.outw[c][p], r.outt[c][p])
            end
        end
    end
    # 最优角度(每任务每 p 每 condition), 画最优 schedule / 检查贴边用。旧版结果没有角度, 跳过。
    if any(r -> hasproperty(r, :outx), RES)
        csv3 = joinpath(directoryBasis, "best_angles_n$(n_at)_inst$(n_done)$(sfx).csv")
        open(csv3, "w") do io
            println(io, "instance,seed,bo_seed,p,condition,ar_cmax,box_scale,durations(t_cost1;t_mix1;t_cost2;...)")
            for r in RES
                hasproperty(r, :outx) || continue
                g = GRAPHS[r.inst]
                for p in 1:p_max, c in CONDS
                    haskey(r.outx[c], p) || continue
                    @printf(io, "%d,%d,%d,%d,%s,%.6f,%g,%s\n", r.inst, g.seed, r.bo_seed, p, c,
                            r.out[c][p], QAOA_BOX_SCALE, join((@sprintf("%.6f", v) for v in r.outx[c][p]), ";"))
                end
            end
        end
        mprintln("✓ 最优角度CSV          → $(basename(csv3))")
    end
    serialize(joinpath(directoryBasis, "batch_qaoa_n$(n_at)_inst$(n_done)_$(optimizer)$(sfx).jls"),
              (GRAPHS=GRAPHS, RES=RES, inst_best=inst_best, inst_bestw=inst_bestw,
               inst_cummax=inst_cummax,
               CONDS=CONDS, p_max=p_max, n_at=n_at, n_bo_seeds=n_bo_seeds))
    mprintln("\n✓ 汇总CSV(每实例best)  → $(basename(csv1))")
    mprintln("✓ 原始CSV(每BO种子)    → $(basename(csv2))")
    mprintln("✓ 完整数据(jls)        → batch_qaoa_n$(n_at)_inst$(n_done)_$(optimizer)$(sfx).jls")
    return nothing
end

# ══════════════════════════════════════════════════════════════════════════
# ── [QNG] 受控对比报表: BO+NelderMead精修 vs BO+QNG精修 ─────────────────────
# 同一批图、同一 BO 种子(= 同样的随机数流、同样的初始设计)、同样的搜索策略
# 与预算, 唯一变量是局部精修器。报三件事:
#   R1 质量(每实例对种子取 best 后配对) —— 论文主表口径, 跟表2一致
#   R2 质量(逐任务配对: 同实例同种子)   —— 更严格, 排除"多种子取best"的平滑
#   R3 成本: 实际量子演化次数(硬件无关) + CPU 时间(仅同一台机器上可比)
#   R4 gnn_warmstart 逐实例明细
# 显著性: 双侧符号检验(精确二项分布, 不假设正态), 外加配对均值 ± 标准误。
# |ΔR_a| < 1e-4 记为持平 —— 这是精修收敛容差的量级, 更小的差异没有物理意义。
# ══════════════════════════════════════════════════════════════════════════
function emit_refiner_comparison(GRAPHS, RES_nm, RES_qng, p_max::Int, n_at::Int,
                                 n_bo_seeds::Int, cfg::String)
    CONDS = ["cold_start", "tqa_init", "gnn_warmstart"]
    tol = 1e-4
    evals_of(r, p, c) = begin
        if !hasproperty(r, :trace) || isempty(r.trace)
            NaN
        else
            sel = [t.n_evals for t in r.trace if t.p == p && t.condition == c]
            isempty(sel) ? NaN : Float64(sum(sel))
        end
    end
    find_r(RS, inst, s) = (i = findfirst(r -> r.inst == inst && r.bo_seed == s, RS);
                           i === nothing ? nothing : RS[i])
    # 精确双侧符号检验 p 值
    sign_p(w, l) = begin
        n = w + l
        n == 0 ? 1.0 :
            min(1.0, 2 * sum(Float64(binomial(BigInt(n), i)) for i in 0:min(w, l)) / 2.0^n)
    end
    se(v) = length(v) > 1 ? std(v) / sqrt(length(v)) : 0.0
    wtl(d) = (count(x -> x > tol, d), count(x -> abs(x) <= tol, d), count(x -> x < -tol, d))
    fmt_nan(x) = isnan(x) ? "   n/a" : @sprintf("%6.0f", x)

    # ── 配对数据: 只保留两种精修器都跑完的 (实例, 种子) ────────────────────
    rows = NamedTuple[]
    for g in GRAPHS, s in 1:n_bo_seeds
        rn = find_r(RES_nm, g.inst, s); rq = find_r(RES_qng, g.inst, s)
        (rn === nothing || rq === nothing) && continue
        for p in 1:p_max, c in CONDS
            push!(rows, (inst=g.inst, gseed=g.seed, bo_seed=s, p=p, cond=c,
                         ar_nm=rn.out[c][p], ar_qng=rq.out[c][p],
                         ev_nm=evals_of(rn, p, c), ev_qng=evals_of(rq, p, c),
                         t_nm=rn.outt[c][p], t_qng=rq.outt[c][p]))
        end
    end
    if isempty(rows)
        mprintln("\n[QNG 对比] 没有两种精修器都完成的任务, 无法出对比表")
        return nothing
    end
    insts = sort(unique(r.inst for r in rows))
    n_pair_tasks = length(unique((r.inst, r.bo_seed) for r in rows))

    # 每实例对种子取 best(只在两边都有的种子上取, 保证同等条件)
    best_nm  = Dict((c, p) => Float64[] for c in CONDS, p in 1:p_max)
    best_qng = Dict((c, p) => Float64[] for c in CONDS, p in 1:p_max)
    for k in insts, c in CONDS, p in 1:p_max
        sel = [r for r in rows if r.inst == k && r.cond == c && r.p == p]
        push!(best_nm[(c, p)],  maximum(r.ar_nm  for r in sel))
        push!(best_qng[(c, p)], maximum(r.ar_qng for r in sel))
    end

    mprintln("\n\n" * "="^78)
    mprintln("【QNG 受控对比】BO+NelderMead精修  vs  BO+QNG精修 (Stokes et al. Quantum 4, 269)")
    mprintln("配对: $(length(insts)) 个实例, $(n_pair_tasks) 个(实例,种子)任务; 其余一切相同, 唯一变量是局部精修器")
    mprintln("QNG 配置: $(QNG_CFG_TAG)   (完整QGT/中心差分h=$(QNG_FD_H), Tikhonov λ=$(QNG_LAM_REL)·tr(g)/d,")
    mprintln("          信赖域 $(QNG_STEP_FRAC)×量程, Armijo回溯, 评估上限=$(QNG_EVAL_FACTOR)×NM迭代上限)")
    mprintln("Δ = QNG − NM (>0 表示 QNG 更好);  |Δ|<$(tol) 记持平;  p_sign = 双侧精确符号检验")
    mprintln("="^78)

    mprintln("\n【表R1】近似比 R_a=<C>/C_max, 每实例先对种子取 best 再配对 (口径同表2)")
    mprintln("-"^78)
    mprintln("  条件           p │  NM mean±std     │  QNG mean±std    │  Δ mean ± SE        │ 赢/平/输  p_sign")
    mprintln("-"^78)
    verdict = Dict{Tuple{String,Int},String}()
    for c in CONDS, p in 1:p_max
        a = best_nm[(c, p)]; b = best_qng[(c, p)]; d = b .- a
        w, t, l = wtl(d); ps = sign_p(w, l)
        @printf("  %-14s %d │ %.4f ± %.4f  │ %.4f ± %.4f  │ %+.5f ± %.5f  │ %2d/%2d/%2d  %.3g\n",
                c, p, mean(a), (length(a)>1 ? std(a) : 0.0), mean(b), (length(b)>1 ? std(b) : 0.0),
                mean(d), se(d), w, t, l, ps)
        verdict[(c, p)] = (ps < 0.05 && w > l && mean(d) > tol) ? "QNG显著更优" :
                          (ps < 0.05 && l > w && mean(d) < -tol) ? "NM显著更优" : "无显著差异"
    end

    mprintln("\n【表R2】逐任务配对(同实例、同BO种子 = 同一随机数流), 更严格的口径")
    mprintln("-"^78)
    mprintln("  条件           p │  Δ mean ± SE        │ 赢/平/输   p_sign  │ max|Δ|")
    mprintln("-"^78)
    for c in CONDS, p in 1:p_max
        d = [r.ar_qng - r.ar_nm for r in rows if r.cond == c && r.p == p]
        w, t, l = wtl(d)
        @printf("  %-14s %d │ %+.5f ± %.5f  │ %2d/%2d/%2d   %.3g  │ %.5f\n",
                c, p, mean(d), se(d), w, t, l, sign_p(w, l), maximum(abs.(d)))
    end

    mprintln("\n【表R3】成本: 每任务实际量子演化次数(硬件无关) 与 CPU 时间")
    mprintln("  [注意] 时间只有两边在同一台机器、同样负载下跑才可比; NM 若来自旧 checkpoint")
    mprintln("         (比如集群上跑的), 请只看演化次数那一列。")
    mprintln("-"^78)
    mprintln("  条件           p │ 演化次数 NM   QNG   QNG/NM │ 分钟 NM    QNG    QNG/NM")
    mprintln("-"^78)
    for c in CONDS, p in 1:p_max
        sel = [r for r in rows if r.cond == c && r.p == p]
        en = [r.ev_nm for r in sel if !isnan(r.ev_nm)]; eq = [r.ev_qng for r in sel if !isnan(r.ev_qng)]
        mn = isempty(en) ? NaN : mean(en); mq = isempty(eq) ? NaN : mean(eq)
        tn = mean(r.t_nm for r in sel)/60; tq = mean(r.t_qng for r in sel)/60
        @printf("  %-14s %d │ %s %s  %s │ %6.2f %6.2f  %6.2f\n",
                c, p, fmt_nan(mn), fmt_nan(mq),
                (isnan(mn) || isnan(mq) || mn == 0) ? "  n/a" : @sprintf("%5.2f", mq/mn),
                tn, tq, (tn > 0 ? tq/tn : NaN))
    end
    let en = [r.ev_nm for r in rows if !isnan(r.ev_nm)], eq = [r.ev_qng for r in rows if !isnan(r.ev_qng)]
        tn = sum(r.t_nm for r in rows)/3600; tq = sum(r.t_qng for r in rows)/3600
        @printf("  合计: 演化次数 NM=%s  QNG=%s   |  CPU NM=%.2f h  QNG=%.2f h\n",
                isempty(en) ? "n/a" : string(round(Int, sum(en))),
                isempty(eq) ? "n/a" : string(round(Int, sum(eq))), tn, tq)
    end

    mprintln("\n【表R4】gnn_warmstart 逐实例明细 (每实例对种子取 best)")
    mprintln("-"^78)
    for (j, k) in enumerate(insts)
        g = GRAPHS[findfirst(gg -> gg.inst == k, GRAPHS)]
        @printf("  实例%2d (图seed=%3d) │", k, g.seed)
        for p in 1:p_max
            a = best_nm[("gnn_warmstart", p)][j]; b = best_qng[("gnn_warmstart", p)][j]
            @printf(" p%d: %.4f→%.4f(%+.4f)", p, a, b, b - a)
        end
        mprintln()
    end

    mprintln("\n【结论】(按表R1口径, 双侧符号检验 α=0.05)")
    for c in CONDS
        @printf("  %-14s: %s\n", c, join(["p$(p) $(verdict[(c, p)])" for p in 1:p_max], " | "))
    end
    n_valid = length(insts)
    n_valid < 6 && mprintln("  [提示] 实例数 $(n_valid) < 6: 符号检验在这个样本量下不可能达到 p<0.05," *
                            " 结论只能看方向和幅度, 想要显著性请加实例。")
    mprintln("="^78)

    mkpath(directoryBasis)
    csvf = joinpath(directoryBasis, "refiner_compare_n$(n_at)_p$(p_max)_$(cfg).csv")
    open(csvf, "w") do io
        println(io, "instance,graph_seed,bo_seed,p,condition,ar_nm,ar_qng,delta,evals_nm,evals_qng,cpu_s_nm,cpu_s_qng")
        for r in rows
            @printf(io, "%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%s,%s,%.1f,%.1f\n",
                    r.inst, r.gseed, r.bo_seed, r.p, r.cond, r.ar_nm, r.ar_qng, r.ar_qng - r.ar_nm,
                    isnan(r.ev_nm) ? "" : string(round(Int, r.ev_nm)),
                    isnan(r.ev_qng) ? "" : string(round(Int, r.ev_qng)), r.t_nm, r.t_qng)
        end
    end
    mprintln("✓ 对比明细CSV → $(basename(csvf))")
    return nothing
end


# ══════════════════════════════════════════════════════════════════════════
# ── 测试实例集: 生成 / 存盘 / 读盘 / 指纹 ───────────────────────────────────
# 为什么需要: 实测同一份代码在 Mac(本地) 和 集群 上生成了**完全不同**的 20 张
# 测试图(连种子 13 这张都不一样: C_max 10.78 vs 10.42)。图的构造链条很长
# (随机拓扑 → 目标权重 → 三维力场嵌入的 LBFGS → "干净"筛选), 任何一环在不同
# Julia 版本/包版本/CPU 上有一点数值差异, 最终的图就不同。结果是两台机器上的
# 数字**不能配对比较**(hybrid vs sweep, NM vs QNG 都一样)。
#
# 解决: 把测试集冻结成一个纯文本文件, 其它机器直接读, 不再重新生成。
#   · 纯文本(不是 .jls): Julia 的 serialize 格式不保证跨版本兼容, 集群 1.11
#     存的 .jls 在 Mac 1.10 上可能读不出来。浮点数用 string(x) 写出, Julia 保证
#     它能精确还原成同一个 Float64(逐比特一致)。
#   · 文件里连 GNN 输出 p_gnn 一起冻结 —— 两台机器的先验若不同, warmstart 的
#     输入也会不同; 冻结后 gnn_warmstart 才是真正的同一个问题。想用本机先验重新
#     推理, 设 QAOA_INSTANCES_REINFER=1 (指纹会随之改变, checkpoint 自动隔离)。
#   · 指纹用 FNV-1a 64 位(不用 Base.hash —— 它不保证跨 Julia 版本稳定), 对
#     每个实例的规范文本计算。两台机器打印出的指纹一样 ⇔ 图逐比特一致。
#
# 用法:
#   生成模式(默认): 照常构造, 并自动存 instances_n12_inst20_<指纹>.txt
#   读盘模式: 设环境变量 QAOA_INSTANCES=<那个 txt 的路径>
# ══════════════════════════════════════════════════════════════════════════
const INSTSET_VERSION = "qaoa-instance-set v1"

"""FNV-1a 64 位哈希, 结果与 Julia 版本/CPU 架构无关。"""
function fnv1a64(s::AbstractString)
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = xor(h, UInt64(b))
        h *= 0x00000100000001b3          # UInt64 乘法自动取模 2^64
    end
    return h
end

"""单个实例的规范文本(也是文件里的存储格式)。只用 string(Float64), 可逐比特还原。"""
function instance_block(g)
    io = IOBuffer()
    println(io, "INSTANCE seed=$(g.seed) n_atoms=$(g.n_atoms) n_edges=$(g.n_edges)")
    println(io, "ATOMS ",   join((string(Float64(c)) for i in 1:g.n_atoms for c in g.atoms[i]), " "))
    println(io, "EDGES ",   join((string(v) for e in g.edges for v in e), " "))
    println(io, "WEIGHTS ", join((string(Float64(w)) for w in g.weights), " "))
    println(io, "PGNN ",    join((string(Float64(p)) for p in g.best_p_gnn), " "))
    # 全部先转 Float64 再写: 若某个量在构造时是 Float32, 写出/读回/再写出仍然
    # 得到同一串字符, 指纹在"生成方"和"读取方"两边才会一致
    F(x) = string(Float64(x))
    println(io, "META cmax=$(F(g.cmax)) total_w=$(F(g.total_w)) unwanted=$(F(g.unwanted)) " *
                "fidelity=$(F(g.fidelity)) stress=$(F(g.stress)) gnn_ar_w=$(F(g.gnn_ar_w))")
    println(io, "END")
    return String(take!(io))
end
instance_fp(g) = string(fnv1a64(instance_block(g)), base=16, pad=16)
set_fp(GR) = string(fnv1a64(join(instance_fp(g) for g in GR)), base=16, pad=16)[1:8]

"""原来 batch / ckpt_report 两处各有一份的构造循环, 抽成一个函数(逻辑逐行不变)。"""
function build_test_instances(n_at::Int, n_instances::Int, gnn_params;
                              wmin::Float64=0.5, wmax::Float64=1.0, verbose::Bool=true)
    GR = NamedTuple[]
    seed_try = 0
    while length(GR) < n_instances && seed_try < n_instances*50
        seed_try += 1
        atoms, edges, weights, stress_val, unwanted_ratio, fidelity =
            make_reg3_weighted_graph_target3d(n_at, seed_try; wmin=wmin, wmax=wmax)
        n_atoms = length(atoms); total_w = sum(weights)
        pos = [atoms[i] for i in 1:n_atoms]
        eset = Set((min(i,j),max(i,j)) for (i,j) in edges)
        ed = Float64[]; ned = Float64[]
        for i in 1:n_atoms, j in (i+1):n_atoms
            xi,yi,zi = pos[i]; xj,yj,zj = pos[j]
            dd = sqrt((xi-xj)^2+(yi-yj)^2+(zi-zj)^2)
            (i,j) in eset ? push!(ed, dd) : push!(ned, dd)
        end
        minimum(ned) <= maximum(ed) && continue     # 嵌入不干净, 跳过
        best_p_gnn, gnn_ar = gnn_infer_weighted(gnn_params, edges, weights, n_atoms)
        cmax = brute_force_maxcut(edges, weights, n_atoms)
        push!(GR, (inst=length(GR)+1, seed=seed_try, atoms=atoms, edges=edges,
                   weights=weights, n_atoms=n_atoms, n_edges=length(edges),
                   total_w=total_w, cmax=cmax, denom=cmax,
                   best_p_gnn=best_p_gnn, gnn_ar_w=gnn_ar, gnn_ar_cmax=gnn_ar*total_w/cmax,
                   unwanted=unwanted_ratio, fidelity=fidelity, stress=stress_val,
                   wmin=minimum(weights), wmax=maximum(weights)))
        verbose && @printf("  实例%2d: seed=%3d  |E|=%2d  C_max=%.4f  C_max/Σw=%.4f  unwanted=%.4f  GNN/C_max=%.4f\n",
                           length(GR), seed_try, length(edges), cmax, cmax/total_w,
                           unwanted_ratio, gnn_ar*total_w/cmax)
    end
    verbose && println("为凑够 $(length(GR)) 个干净实例, 共试了 $(seed_try) 个种子")
    return GR
end

function save_instance_set(path::String, GR; note::String="")
    open(path, "w") do io
        println(io, "# ", INSTSET_VERSION)
        println(io, "# fingerprint=$(set_fp(GR)) n_instances=$(length(GR)) n_atoms=$(GR[1].n_atoms)")
        println(io, "# created=$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) host=$(gethostname()) " *
                    "julia=$(VERSION) $(note)")
        println(io, "# 每个实例: INSTANCE/ATOMS(x y z ...)/EDGES(u v ...)/WEIGHTS/PGNN/META/END")
        for g in GR
            print(io, instance_block(g))
        end
    end
    return path
end

"""读实例集文件。取前 n_instances 个; 做完整性校验(权重与原子位置一致、
   C_max 暴力枚举复核), 任何一项对不上就报错 —— 宁可停下也不要用错的图。"""
function load_instance_set(path::String, n_at::Int, n_instances::Int)
    isfile(path) || error("QAOA_INSTANCES 指向的文件不存在: $(path)")
    lines = readlines(path)
    (isempty(lines) || !startswith(lines[1], "# " * INSTSET_VERSION)) &&
        error("不是实例集文件(或版本不对): $(path)\n  第一行应为 \"# $(INSTSET_VERSION)\"")
    kv(s) = Dict(String(first(split(t, "=", limit=2))) => String(last(split(t, "=", limit=2)))
                 for t in split(s))
    GR = NamedTuple[]; cur = Dict{String,String}()
    for ln in lines
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        k = findfirst(' ', ln)
        key  = k === nothing ? ln : ln[1:k-1]
        rest = k === nothing ? "" : ln[k+1:end]
        if key == "INSTANCE"
            cur = Dict("INSTANCE" => rest)
        elseif key == "END"
            h = kv(cur["INSTANCE"]); m = kv(cur["META"])
            n = parse(Int, h["n_atoms"])
            xs = parse.(Float64, split(cur["ATOMS"]))
            length(xs) == 3n || error("实例 seed=$(h["seed"]): ATOMS 长度 $(length(xs)) ≠ 3×$(n)")
            positions = [(xs[3i-2], xs[3i-1], xs[3i]) for i in 1:n]
            evs = parse.(Int, split(cur["EDGES"]))
            edges = Tuple{Int,Int}[(evs[2j-1], evs[2j]) for j in 1:(length(evs) ÷ 2)]
            weights = parse.(Float64, split(cur["WEIGHTS"]))
            pg = parse.(Float64, split(cur["PGNN"]))
            length(weights) == length(edges) || error("实例 seed=$(h["seed"]): 权重数与边数不符")
            length(pg) == n || error("实例 seed=$(h["seed"]): PGNN 长度 ≠ n_atoms")
            total_w = sum(weights); cmax = parse(Float64, m["cmax"])
            gnn_ar_w = parse(Float64, m["gnn_ar_w"])
            push!(GR, (inst=length(GR)+1, seed=parse(Int, h["seed"]), atoms=AtomList(positions),
                       edges=edges, weights=weights, n_atoms=n, n_edges=length(edges),
                       total_w=total_w, cmax=cmax, denom=cmax,
                       best_p_gnn=pg, gnn_ar_w=gnn_ar_w, gnn_ar_cmax=gnn_ar_w*total_w/cmax,
                       unwanted=parse(Float64, m["unwanted"]), fidelity=parse(Float64, m["fidelity"]),
                       stress=parse(Float64, m["stress"]),
                       wmin=minimum(weights), wmax=maximum(weights)))
            cur = Dict{String,String}()
        else
            cur[key] = rest
        end
    end
    isempty(GR) && error("实例集文件里没有读到任何实例: $(path)")
    GR[1].n_atoms == n_at ||
        error("实例集是 n_atoms=$(GR[1].n_atoms) 的, 但命令里给的是 n_atoms=$(n_at)")
    length(GR) >= n_instances ||
        error("实例集只有 $(length(GR)) 个实例, 命令要求 $(n_instances) 个")
    GR = GR[1:n_instances]
    # ── 完整性校验 ─────────────────────────────────────────────────────
    for g in GR
        # (1) 权重必须等于由原子位置算出的范德华耦合 (构造时就是这么定义的)
        for (idx, (i, j)) in enumerate(g.edges)
            xi,yi,zi = g.atoms[i]; xj,yj,zj = g.atoms[j]
            r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
            wv = C6_VAL/(4.0*r2^3)
            abs(wv - g.weights[idx]) <= 1e-9*max(1.0, abs(wv)) ||
                error("实例 seed=$(g.seed) 边$(idx): 权重 $(g.weights[idx]) 与原子位置算出的 $(wv) 不一致, 文件可能被改动过")
        end
        # (2) C_max 暴力枚举复核 (n=12 只要毫秒级)
        cm = brute_force_maxcut(g.edges, g.weights, g.n_atoms)
        abs(cm - g.cmax) <= 1e-9*max(1.0, abs(cm)) ||
            error("实例 seed=$(g.seed): 文件里 C_max=$(g.cmax), 复算得 $(cm)")
    end
    return GR
end

"""跟当前先验的推理结果对比; reinfer=true 时用当前先验替换文件里冻结的 p_gnn。"""
function reconcile_gnn(GR, gnn_params; reinfer::Bool=false)
    dmax = 0.0; n_flip = 0; out = NamedTuple[]
    for g in GR
        p_now, ar_now = gnn_infer_weighted(gnn_params, g.edges, g.weights, g.n_atoms)
        dmax = max(dmax, maximum(abs.(p_now .- g.best_p_gnn)))
        n_flip += count((p_now .> 0.5) .!= (g.best_p_gnn .> 0.5))
        push!(out, reinfer ? merge(g, (best_p_gnn=p_now, gnn_ar_w=ar_now,
                                       gnn_ar_cmax=ar_now*g.total_w/g.cmax)) : g)
    end
    @printf("  本机先验 vs 文件里冻结的 GNN 输出: max|Δp|=%.2e, 硬判决(p>0.5)不同的原子共 %d 个\n",
            dmax, n_flip)
    if reinfer
        println("  → QAOA_INSTANCES_REINFER=1: 改用本机先验的推理结果(指纹随之改变)")
    elseif dmax > 1e-6
        println("  → 两边先验不同。默认仍用文件里冻结的值(保证 warmstart 输入跟生成方一致);")
        println("    想用本机先验, 设 QAOA_INSTANCES_REINFER=1")
    end
    return out
end

"""checkpoint 目录与实例集的对应关系校验。原来 checkpoint 目录名只编码
   n/p/种子数/配置, **不含图本身** —— 换了一批图(比如读了别的机器的实例集)
   也会静默加载旧任务, 把两批图的结果混在一起。现在每个目录里记录每个实例
   的指纹, 对不上就停。允许"前缀复用": 6 个实例的目录可以被 20 个实例的运行
   接着用(前 6 个指纹一致即可)。"""
function check_ckpt_fingerprints(dir::String, GR, fps::Vector{String};
                                 from_file::Bool, write::Bool=true)
    f = joinpath(dir, "instances_fingerprint.txt")
    has_tasks = isdir(dir) && any(startswith(x, "task_i") for x in readdir(dir))
    rec = Dict{Int,String}()
    if isfile(f)
        for ln in readlines(f)
            m = match(r"^inst=(\d+) fp=([0-9a-f]+)", ln)
            m === nothing || (rec[parse(Int, m[1])] = String(m[2]))
        end
        bad = [g.inst for (g, fp) in zip(GR, fps) if haskey(rec, g.inst) && rec[g.inst] != fp]
        if !isempty(bad)
            println("="^70)
            println("checkpoint 目录 $(basename(dir)) 里的任务是在**另一批图**上算的!")
            println("  指纹不一致的实例: $(bad)")
            println("  直接续跑会把两批图的结果混在一起, 所以这里停下。可选:")
            println("   · 换一个输出目录: 设 QAOA_OUTDIR=<新目录>")
            println("   · 或确认旧结果不要了, 删掉这个 checkpoint 目录")
            println("="^70)
            error("checkpoint 与当前实例集不匹配, 中止")
        end
        n_ok = count(g -> haskey(rec, g.inst), GR)
        println("  ✓ $(basename(dir)): 实例指纹校验通过 ($(n_ok) 个已记录, $(length(GR)-n_ok) 个新增)")
    elseif has_tasks
        if from_file && get(ENV, "QAOA_TRUST_LEGACY_CKPT", "") != "1"
            println("="^70)
            println("checkpoint 目录 $(basename(dir)) 是旧版本留下的(没有实例指纹),")
            println("无法确认它是不是在当前读入的这批图上算的。")
            println("  · 如果它就是生成这个实例集文件的那台机器上跑的(同一批图), 设")
            println("    QAOA_TRUST_LEGACY_CKPT=1 继续复用")
            println("  · 否则换输出目录(QAOA_OUTDIR=<新目录>)或删掉这个目录")
            println("="^70)
            error("旧 checkpoint 无法校验, 中止")
        end
        println("  ⚠ $(basename(dir)): 旧版 checkpoint(无指纹), 按旧规则默认它属于当前这批图;")
        println("    从现在起写入指纹, 以后换图会被自动拦下。")
    end
    if write
        newrec = [(g.inst, fp, g.seed) for (g, fp) in zip(GR, fps) if !haskey(rec, g.inst)]
        if !isempty(newrec)
            mkpath(dir)
            open(f, "a") do io
                for (k, fp, sd) in newrec
                    println(io, "inst=$(k) fp=$(fp) seed=$(sd)")
                end
            end
        end
    end
    return nothing
end

"""批量模式和 ckpt_report 共用的入口: 读盘或生成 → 打印指纹 → (生成时)存盘。
   返回 (GRAPHS, 每实例指纹, 整体指纹, 是否来自文件)。"""
function obtain_test_instances(n_at::Int, n_instances::Int, gnn_params;
                               wmin::Float64=0.5, wmax::Float64=1.0,
                               verbose::Bool=true, save::Bool=true, prior_tag::String="")
    inst_file = get(ENV, "QAOA_INSTANCES", "")
    from_file = !isempty(inst_file)
    GR = if from_file
        println("\n从实例集文件读取 $(n_instances) 个测试实例(不重新生成): $(inst_file)")
        g0 = load_instance_set(inst_file, n_at, n_instances)
        println("  ✓ 完整性校验通过(权重↔原子位置一致, C_max 暴力复核一致)")
        g1 = reconcile_gnn(g0, gnn_params; reinfer = get(ENV, "QAOA_INSTANCES_REINFER", "") == "1")
        if verbose
            for g in g1
                @printf("  实例%2d: seed=%3d  |E|=%2d  C_max=%.4f  C_max/Σw=%.4f  unwanted=%.4f  GNN/C_max=%.4f\n",
                        g.inst, g.seed, g.n_edges, g.cmax, g.cmax/g.total_w, g.unwanted, g.gnn_ar_cmax)
            end
        end
        g1
    else
        verbose && println("\n构建 $(n_instances) 个测试实例...")
        build_test_instances(n_at, n_instances, gnn_params; wmin=wmin, wmax=wmax, verbose=verbose)
    end
    fps = [instance_fp(g) for g in GR]
    sfp = set_fp(GR)
    println("测试实例集指纹: $(sfp)   (两台机器指纹相同 ⇔ 图与 GNN 输入逐比特一致)")
    if !from_file && save
        mkpath(directoryBasis)
        pth = joinpath(directoryBasis, "instances_n$(n_at)_inst$(length(GR))_$(sfp).txt")
        if isfile(pth)
            println("  实例集文件已存在(内容由指纹保证一致): $(basename(pth))")
        else
            save_instance_set(pth, GR; note="prior=$(prior_tag)")
            println("  ✓ 实例集已冻结 → $(basename(pth))")
        end
        println("    在别的机器上复用这批图: 把它拷过去并设 QAOA_INSTANCES=<路径>")
    end
    return GR, fps, sfp, from_file
end

# ══════════════════════════════════════════════════════════════════════════
# ── kernel_ablation: 核消融实验 ────────────────────────────────────────────
# 目的: 回答"GSM+CSM 核到底贡献了什么" —— 生产流程里约 90% 的演化来自 NM
# 精修, 最终结果又已被网格证明到顶, 所以从最终 R_a 看不出核的作用。这里把
# NM 精修和停滞重启**全部关掉**, 只比较"纯 BO"用同样的预算能多快逼近最优:
#   gsmcsm_v2 : GSM+CSM 核 + GP v2(y 标准化/输入归一化/联合TS/加强超参数拟合)
#   gsmcsm_v2hp: 同 gsmcsm_v2, 但超参数拟合迭代次数按参数个数放大(高维时公平对待谱混合核)
#   gsmcsm_v1 : GSM+CSM 核 + 原版 GP(= 生产流程里的 GP, 只是去掉了 NM)
#   matern52  : ARD Matérn-5/2 核(BO 领域的标准默认核), 同样用 v2 约定
#   rbf       : ARD 平方指数核, 同样用 v2 约定
#   random    : 均匀随机采样(不用 GP 的下限基线)
# 所有方法: 同一个随机种子 ⇒ **完全相同的初始设计**(配对比较); 同样的采集
# 函数日程(TS → EI → LCB, 与生产 2D BO 一致)、同样的候选点数、同样的重拟合
# 频率。基线核也享受 v2 的全部改进, 比较只针对"核"本身, 对 GSM+CSM 不偏袒。
#
# 标准答案: p=1 用 n_grid×n_grid 网格 + 3 个起点的 NM 精修(已验证可信);
#           p≥2 用"所有方法所有重复里的最好值", 若设 QAOA_ABL_BASELINE=<基准
#           checkpoint 目录名> 则再并入该基准(比如 hybrid)的结果。
# p≥2 时是**不锁前缀**的联合 2p 维 BO, 顺带回答"放开参数空间 GP 能否胜过逐层
# 策略": 表 C 直接给出联合 BO 与基准(逐层 hybrid)在同一 p 上的配对比较。
#
# 运行方式:
#   julia r3g_sge_integrated.jl kernel_ablation <n_atoms> <n_instances> <p列表> <预算> <重复数> [方法列表] [网格分辨率]
# 例如:
#   ... kernel_ablation 12 20 1 300 3                         (p=1, 默认 5 种方法)
#   ... kernel_ablation 12 20 2 400 2 gsmcsm_v2,matern52,random
# 建议始终设 QAOA_INSTANCES=<实例集文件>, 保证与基准同一批图。
# ══════════════════════════════════════════════════════════════════════════
@everywhere begin
const ABL_METHODS = (:gsmcsm_v2, :gsmcsm_v2ms, :gsmcsm_v2hp, :gsmcsm_v1, :matern52, :rbf, :random,
                     :gsmcsm_paper, :matern52_paper)
# gsmcsm_paper / matern52_paper: 论文协议(见 pp_run_bo 上方说明), 两者只差核。
# gsmcsm_v2ms: 同 gsmcsm_v2, 但超参数拟合用 8 次重启(v2 为 3 次), 且每次 NM 停下后再从该点
#   重启单纯形跑一遍。p=3/4 实测 gsmcsm_v2hp(只放大迭代上限)与 v2 **逐比特相同** —— 说明
#   NM 根本没用到 800 次上限就按收敛判据停了, 放大上限是无效旋钮; 重启次数和单纯形重启
#   才是真正增加拟合力度的办法。拟合开销约为 v2 的 5 倍。
const ABL_MS_RESTARTS = 8
# gsmcsm_v2hp: 与 gsmcsm_v2 完全相同, 只是超参数拟合迭代次数按参数个数放大
#   (max(800, 40×参数个数); 6 维 52 个参数 → 2080 次, 8 维 68 个 → 2720 次)。
#   用来区分"谱混合核本身不行"和"只是我们的超参数没拟合到位": 如果放大后仍然
#   明显输给 Matérn, 问题就在核本身。
abl_hp_iters(D::Int) = max(GP_V2_HP_ITERS, 40 * 4 * (1 + 2D))

"""ARD 平稳核(输入已归一化)。θ = [log σ², log ℓ_1..ℓ_D]。"""
function abl_kernel(kind::Symbol, X1::Matrix{Float64}, X2::Matrix{Float64},
                    θ::Vector{Float64}, D::Int)
    s2 = exp(θ[1]); r2 = zeros(size(X1,2), size(X2,2))
    for d in 1:D
        Δ = X1[d:d,:]' .- X2[d:d,:]
        r2 .+= (Δ ./ exp(θ[1+d])).^2
    end
    if kind == :rbf
        return s2 .* exp.(-0.5 .* r2)
    else                                   # :matern52
        r = sqrt.(r2)
        return s2 .* (1 .+ sqrt(5) .* r .+ (5/3) .* r2) .* exp.(-sqrt(5) .* r)
    end
end

struct AblGP
    kind::Symbol; L::LowerTriangular{Float64,Matrix{Float64}}; α::Vector{Float64}
    X::Matrix{Float64}; θ::Vector{Float64}; D::Int; ym::Float64; ysd::Float64; xs::Vector{Float64}
end

"""基线核的 MAP 超参数(先验 N(0,1), 与 GSM+CSM 相同; 迭代/重启次数与 GP v2 相同)。"""
function abl_fit(kind::Symbol, X::Matrix{Float64}, y::Vector{Float64}, D::Int,
                 xs::Vector{Float64}; θ_init=nothing, nr::Int=2)
    Xn = X ./ xs; ym, sd = _gp_ystd(y); yz = (y .- ym) ./ sd
    n_p = 1 + D; lo = -6.0*ones(n_p); hi = 6.0*ones(n_p)
    function nll(θ)
        try
            L, _ = safe_chol(abl_kernel(kind, Xn, Xn, θ, D), 1e-4)
            α = L' \ (L \ yz)
            v = 0.5*dot(yz, α) + sum(log.(diag(L))) + 0.5*length(yz)*log(2π) + 0.5*sum(θ.^2)
            return isfinite(v) ? v : 1e10
        catch
            return 1e10
        end
    end
    bθ, bv = nothing, Inf
    for ri in 1:(nr + GP_V2_EXTRA_RESTARTS)
        θ0 = (ri == 1 && θ_init !== nothing && length(θ_init) == n_p) ?
             clamp.(θ_init, lo .+ 1e-6, hi .- 1e-6) : lo .+ rand(n_p) .* (hi .- lo)
        r = optimize(nll, lo, hi, θ0, Fminbox(NelderMead()),
                     Optim.Options(iterations=GP_V2_HP_ITERS, show_trace=false))
        Optim.minimum(r) < bv && (bv = Optim.minimum(r); bθ = Optim.minimizer(r))
    end
    return bθ === nothing ? zeros(n_p) : bθ
end
function abl_build(kind::Symbol, X::Matrix{Float64}, y::Vector{Float64}, θ::Vector{Float64},
                   D::Int, xs::Vector{Float64})
    Xn = X ./ xs; ym, sd = _gp_ystd(y); yz = (y .- ym) ./ sd
    L, _ = safe_chol(abl_kernel(kind, Xn, Xn, θ, D), 1e-4)
    return AblGP(kind, L, L' \ (L \ yz), Xn, θ, D, ym, sd, xs)
end
function abl_pred(gp::AblGP, C::Matrix{Float64})
    Kx = abl_kernel(gp.kind, gp.X, C ./ gp.xs, gp.θ, gp.D); v = gp.L \ Kx
    μ = Kx' * gp.α
    σ = sqrt.(max.(exp(gp.θ[1]) .- vec(sum(v.^2; dims=1)), 0.0))
    return gp.ym .+ gp.ysd .* μ, gp.ysd .* σ              # 原始单位
end
function abl_ts(gp::AblGP, C::Matrix{Float64}; ns::Int=10)
    m = size(C,2); sub = m > TS_JOINT_MAX ? randperm(m)[1:TS_JOINT_MAX] : collect(1:m)
    Cq = (C ./ gp.xs)[:, sub]
    Kx = abl_kernel(gp.kind, gp.X, Cq, gp.θ, gp.D); v = gp.L \ Kx; μ = Kx' * gp.α
    Σ = abl_kernel(gp.kind, Cq, Cq, gp.θ, gp.D) .- v' * v; Σ = (Σ .+ Σ') ./ 2
    Lc, _ = safe_chol(Σ, 1e-8*max(exp(gp.θ[1]), 1e-12))
    bi = 0; bs = Inf
    for _ in 1:ns
        smp = μ .+ Lc * randn(length(sub)); k = argmin(smp)
        smp[k] < bs && (bs = smp[k]; bi = sub[k])
    end
    return bi
end

"""纯 BO(无 NM 精修、无停滞重启)。返回每次评估后的 best-so-far 曲线(原始损失单位)。"""
function abl_run_bo(evf, lo::Vector{Float64}, hi::Vector{Float64}, method::Symbol,
                    budget::Int, seed::Int)
    D = length(lo); xs = hi .- lo
    # ── 共同初始设计: 只依赖 seed, 与方法无关 ──
    Random.seed!(seed)
    n_init = D <= 2 ? min(24, max(12, budget ÷ 8)) : min(40, max(20, budget ÷ 6))
    n_init = min(n_init, budget)
    H = halton_samples(max(n_init-1, 1), lo, hi; skip=7)
    init = Vector{Vector{Float64}}([zeros(D)])            # 零时长 = 只做态制备
    for i in 1:(n_init-1)
        push!(init, clamp.(H[i] .+ randn(D) .* 0.05 .* xs, lo, hi))
    end
    t0 = time(); tfit = 0.0
    X = zeros(D, 0); ys = Float64[]; curve = Float64[]; by = Inf; bx = zeros(D)
    for u in init
        yv = evf(u); X = hcat(X, u); push!(ys, yv)
        yv < by && (by = yv; bx = copy(u))
        push!(curve, by)
    end
    # ── 方法各自的随机流从这里开始, 不影响上面的共同设计 ──
    Random.seed!(seed + 7919)
    n_iter = budget - n_init
    n_cand = D <= 2 ? 400 : 2000
    Q_g = 2; Q_c = 2
    local θ
    θ = nothing
    for it in 1:n_iter
        local xn
        if method == :random
            xn = lo .+ rand(D) .* xs
        else
            t = it / n_iter
            if θ === nothing || it % 18 == 0                  # 与生产 2D BO 相同的重拟合频率
                tf = time()
                θ = if method == :gsmcsm_v2
                    map_spectral_hp(X, ys, Q_g, Q_c, D; nr=2, θ_init=θ, gpv=2, xscale=xs)
                elseif method == :gsmcsm_v2hp
                    map_spectral_hp(X, ys, Q_g, Q_c, D; nr=2, θ_init=θ, gpv=2, xscale=xs,
                                    hp_iters=abl_hp_iters(D))
                elseif method == :gsmcsm_v2ms
                    map_spectral_hp(X, ys, Q_g, Q_c, D; nr=2, θ_init=θ, gpv=2, xscale=xs,
                                    hp_restarts=ABL_MS_RESTARTS, hp_rerun=true)
                elseif method == :gsmcsm_v1
                    map_spectral_hp(X, ys, Q_g, Q_c, D; nr=2, θ_init=θ, gpv=1)
                else
                    abl_fit(method, X, ys, D, xs; θ_init=θ)
                end
                tfit += time() - tf
            end
            C = if D <= 2
                lo .+ rand(D, n_cand) .* xs
            else                                              # 高维: 80% 全局均匀 + 20% 当前最优附近
                nl = n_cand ÷ 5
                hcat(lo .+ rand(D, n_cand - nl) .* xs,
                     clamp.(bx .+ randn(D, nl) .* 0.05 .* xs, lo, hi))
            end
            if method in (:gsmcsm_v2, :gsmcsm_v2hp, :gsmcsm_v2ms, :gsmcsm_v1)
                gv = method == :gsmcsm_v1 ? 1 : 2
                gp = gv == 2 ? build_spectral_gp(X, ys, θ, Q_g, Q_c, D; gpv=2, xscale=xs) :
                               build_spectral_gp(X, ys, θ, Q_g, Q_c, D; gpv=1)
                xn = if t < 0.30
                    C[:, ts_acq(gp, C; ns=10)]
                elseif t < 0.68
                    μ, σ = gp_pred(gp, C)
                    C[:, argmax(ei_acq(μ, σ, by; ξ=0.05*exp(-3.0*(t-0.30)/0.38)))]
                else
                    μ, σ = gp_pred(gp, C)
                    C[:, argmin(lcb_acq(μ, σ; β=max(exp(-2.5*(t-0.68)/0.32), 0.05)))]
                end
            else
                ga = abl_build(method, X, ys, θ, D, xs)
                xn = if t < 0.30
                    C[:, abl_ts(ga, C; ns=10)]
                elseif t < 0.68
                    μ, σ = abl_pred(ga, C)
                    C[:, argmax(ei_acq(μ, σ, by; ξ=0.05*exp(-3.0*(t-0.30)/0.38)))]
                else
                    μ, σ = abl_pred(ga, C)
                    C[:, argmin(lcb_acq(μ, σ; β=max(exp(-2.5*(t-0.68)/0.32), 0.05)))]
                end
            end
        end
        yv = evf(xn); X = hcat(X, xn); push!(ys, yv)
        yv < by && (by = yv; bx = copy(xn))
        push!(curve, by)
    end
    return (curve=curve, ybest=by, xbest=bx, t_total=time()-t0, t_fit=tfit, n_init=n_init)
end

# ── [论文协议] 尽量贴近 Zhang & Hua (arXiv:2505.17393) 的 BO 设置 ──────────────
# 与本文件其它 BO 的区别(全部按论文 / BoTorch 惯例):
#   · 核: CSM+GSM = 6 个 Cauchy + 1 个 Gaussian 成分(论文主文脚注 3 的配置)
#   · 观测噪声 σ_n² 作为超参数由边际似然拟合(论文附录 C.1: "kernel hyperparameters,
#     as well as the observation noise, are inferred via marginal likelihood
#     maximization after each function evaluation")
#   · **每次评估后**都重新拟合超参数; 纯边际似然(不加先验), 基于梯度(L-BFGS,
#     有限差分梯度)从上一次的解热启动, 每 PP_RESTART_EVERY 次再加一个随机重启
#   · 采集函数 UCB(最小化即 LCB), β_t = 2 log(|X| t² π² / 6δ)(论文命题 5.2 / Srinivas),
#     |X| 取候选点数, δ = 0.1; 采集函数用随机候选 + L-BFGS 连续优化(论文附录 C.1)
#   · 输入归一化到 [0,1]、输出标准化(BoTorch 默认做法)
# matern52_paper 用**完全相同**的协议, 只把核换成 Matérn-5/2 —— 这样两者的差别只来自核。
const PP_QG = 1
const PP_QC = 6
const PP_RESTART_EVERY = 10
const PP_DELTA = 0.1

pp_nkern(kind::Symbol, D::Int) = kind == :gsmcsm ? (PP_QG + PP_QC) * (1 + 2D) : 1 + D
function pp_kern(kind::Symbol, X1::Matrix{Float64}, X2::Matrix{Float64}, θk::Vector{Float64}, D::Int)
    kind == :gsmcsm ? spectral_mixture_kernel(X1, X2, θk, PP_QG, PP_QC, D) :
                      abl_kernel(:matern52, X1, X2, θk, D)
end
pp_priorvar(kind::Symbol, θk::Vector{Float64}, D::Int) =
    kind == :gsmcsm ? _prior_var(θk, PP_QG, PP_QC, D) : exp(θk[1])
function pp_bounds(kind::Symbol, D::Int, xs::Vector{Float64})
    lo, hi = if kind == :gsmcsm
        l, h, _ = _spectral_bounds(PP_QG, PP_QC, D, xs); (l, h)
    else
        (-6.0*ones(1+D), 6.0*ones(1+D))
    end
    return vcat(lo, -14.0), vcat(hi, 0.0)              # 最后一维: log σ_n²(标准化单位)
end

struct PPGP
    kind::Symbol; L::LowerTriangular{Float64,Matrix{Float64}}; α::Vector{Float64}
    X::Matrix{Float64}; θ::Vector{Float64}; D::Int; ym::Float64; ysd::Float64; xs::Vector{Float64}
end

"""纯边际似然拟合(含噪声)。θ_init 给定时从它热启动(L-BFGS); n_rand 个额外随机起点。"""
function pp_fit(kind::Symbol, X::Matrix{Float64}, y::Vector{Float64}, D::Int, xs::Vector{Float64};
                θ_init=nothing, n_rand::Int=0)
    Xn = X ./ xs; ym, sd = _gp_ystd(y); yz = (y .- ym) ./ sd
    lo, hi = pp_bounds(kind, D, xs); nk = length(lo) - 1
    function nll(θ)
        try
            K = pp_kern(kind, Xn, Xn, θ[1:nk], D)
            K[diagind(K)] .+= exp(θ[end]) + 1e-8
            L, _ = safe_chol(K, 1e-8)
            α = L' \ (L \ yz)
            v = 0.5*dot(yz, α) + sum(log.(diag(L))) + 0.5*length(yz)*log(2π)
            return isfinite(v) ? v : 1e10
        catch
            return 1e10
        end
    end
    starts = Vector{Vector{Float64}}()
    θ_init !== nothing && length(θ_init) == length(lo) && push!(starts, clamp.(θ_init, lo .+ 1e-6, hi .- 1e-6))
    for _ in 1:(n_rand + (isempty(starts) ? 1 : 0))
        push!(starts, lo .+ (0.05 .+ 0.9 .* rand(length(lo))) .* (hi .- lo))
    end
    bθ = starts[1]; bv = nll(bθ)
    for θ0 in starts
        r = try
            optimize(nll, lo, hi, θ0, Fminbox(LBFGS()),
                     Optim.Options(iterations=15, outer_iterations=2, show_trace=false))
        catch
            optimize(nll, lo, hi, θ0, Fminbox(NelderMead()), Optim.Options(iterations=400))
        end
        Optim.minimum(r) < bv && (bv = Optim.minimum(r); bθ = Optim.minimizer(r))
    end
    return bθ
end
function pp_build(kind::Symbol, X::Matrix{Float64}, y::Vector{Float64}, θ::Vector{Float64},
                  D::Int, xs::Vector{Float64})
    Xn = X ./ xs; ym, sd = _gp_ystd(y); yz = (y .- ym) ./ sd; nk = length(θ) - 1
    K = pp_kern(kind, Xn, Xn, θ[1:nk], D); K[diagind(K)] .+= exp(θ[end]) + 1e-8
    L, _ = safe_chol(K, 1e-8)
    return PPGP(kind, L, L' \ (L \ yz), Xn, θ, D, ym, sd, xs)
end
"""潜函数 f 的后验(不含观测噪声), 原始单位。"""
function pp_pred(gp::PPGP, C::Matrix{Float64})
    nk = length(gp.θ) - 1
    Kx = pp_kern(gp.kind, gp.X, C ./ gp.xs, gp.θ[1:nk], gp.D); v = gp.L \ Kx
    μ = Kx' * gp.α
    σ = sqrt.(max.(pp_priorvar(gp.kind, gp.θ[1:nk], gp.D) .- vec(sum(v.^2; dims=1)), 0.0))
    return gp.ym .+ gp.ysd .* μ, gp.ysd .* σ
end

"""论文协议的 BO 主循环; 初始设计与 abl_run_bo 完全相同(同一 seed ⇒ 同一批初始点)。"""
function pp_run_bo(evf, lo::Vector{Float64}, hi::Vector{Float64}, kind::Symbol,
                   budget::Int, seed::Int)
    D = length(lo); xs = hi .- lo
    Random.seed!(seed)
    n_init = D <= 2 ? min(24, max(12, budget ÷ 8)) : min(40, max(20, budget ÷ 6))
    n_init = min(n_init, budget)
    H = halton_samples(max(n_init-1, 1), lo, hi; skip=7)
    init = Vector{Vector{Float64}}([zeros(D)])
    for i in 1:(n_init-1)
        push!(init, clamp.(H[i] .+ randn(D) .* 0.05 .* xs, lo, hi))
    end
    t0 = time(); tfit = 0.0
    X = zeros(D, 0); ys = Float64[]; curve = Float64[]; by = Inf; bx = zeros(D)
    for u in init
        yv = evf(u); X = hcat(X, u); push!(ys, yv)
        yv < by && (by = yv; bx = copy(u)); push!(curve, by)
    end
    Random.seed!(seed + 7919)
    n_cand = D <= 2 ? 1000 : 2000
    local θ
    θ = nothing
    for it in 1:(budget - n_init)
        tf = time()
        θ = pp_fit(kind, X, ys, D, xs; θ_init=θ,
                   n_rand = (θ === nothing ? 3 : (it % PP_RESTART_EVERY == 0 ? 1 : 0)))
        tfit += time() - tf
        gp = pp_build(kind, X, ys, θ, D, xs)
        βt = 2.0 * log(n_cand * it^2 * π^2 / (6.0 * PP_DELTA))
        acq = u -> begin
            μu, σu = pp_pred(gp, reshape(clamp.(u, lo, hi), D, 1))
            μu[1] - sqrt(βt) * σu[1]
        end
        C = lo .+ rand(D, n_cand) .* xs
        μc, σc = pp_pred(gp, C); a = μc .- sqrt(βt) .* σc
        order = sortperm(a)
        xn = C[:, order[1]]; an = a[order[1]]
        for j in order[1:min(3, length(order))]                # 从前 3 个候选点出发连续优化采集函数
            r = try
                optimize(acq, lo, hi, clamp.(C[:, j], lo .+ 1e-9, hi .- 1e-9), Fminbox(LBFGS()),
                         Optim.Options(iterations=20, outer_iterations=2))
            catch
                nothing
            end
            r === nothing && continue
            Optim.minimum(r) < an && (an = Optim.minimum(r); xn = clamp.(Optim.minimizer(r), lo, hi))
        end
        yv = evf(xn); X = hcat(X, xn); push!(ys, yv)
        yv < by && (by = yv; bx = copy(xn)); push!(curve, by)
    end
    return (curve=curve, ybest=by, xbest=bx, t_total=time()-t0, t_fit=tfit, n_init=n_init)
end

"""p=1 标准答案: 网格 + 从网格前 3 个互相分开的点各做一次 NM 精修。"""
function abl_grid_truth(evf, lo::Vector{Float64}, hi::Vector{Float64}, n_grid::Int)
    t0 = time()
    vals = Tuple{Float64,Vector{Float64}}[]
    for γ in range(lo[1], hi[1], length=n_grid), β in range(lo[2], hi[2], length=n_grid)
        push!(vals, (evf([γ, β]), [γ, β]))
    end
    sort!(vals, by = first)
    ygrid = vals[1][1]
    starts = Vector{Vector{Float64}}()
    for (_, x) in vals
        all(s0 -> maximum(abs.(s0 .- x) ./ (hi .- lo)) > 0.05, starts) && push!(starts, x)
        length(starts) >= 3 && break
    end
    by = ygrid; bx = vals[1][2]
    for x0 in starts
        r = optimize(u -> evf(clamp.(u, lo, hi)), x0, NelderMead(),
                     Optim.Options(iterations=400, f_reltol=1e-10))
        Optim.minimum(r) < by && (by = Optim.minimum(r); bx = clamp.(Optim.minimizer(r), lo, hi))
    end
    return (ybest=by, xbest=bx, ygrid=ygrid, t_total=time()-t0)
end

function abl_task(tk)
    g = tk.g; cnd = tk.cond; p = tk.p
    lo = zeros(2p); hi = vcat([[BOX_HI[1], BOX_HI[2]] for _ in 1:p]...)
    evf = x -> begin
        xq = clamp.(x, lo, hi)
        try
            lv = cnd == "gnn_warmstart" ?
                 loss_fn_warmstart_weighted(xq, g.best_p_gnn, g.atoms, g.edges, g.weights, g.n_atoms) :
                 loss_fn_weighted(xq, g.atoms, g.edges, g.weights, g.n_atoms)
            isfinite(lv) ? clamp(lv, -(g.total_w+1.0), 1000.0) : 1000.0
        catch
            1000.0
        end
    end
    if tk.method == :grid
        r = abl_grid_truth(evf, lo, hi, tk.n_grid)
        return (inst=g.inst, cond=cnd, p=p, method=:grid, rep=0, curve=Float64[],
                ybest=r.ybest, ygrid=r.ygrid, xbest=r.xbest, t_total=r.t_total, t_fit=0.0, n_init=0)
    end
    r = tk.method == :gsmcsm_paper   ? pp_run_bo(evf, lo, hi, :gsmcsm,   tk.budget, tk.seed) :
        tk.method == :matern52_paper ? pp_run_bo(evf, lo, hi, :matern52, tk.budget, tk.seed) :
                                       abl_run_bo(evf, lo, hi, tk.method, tk.budget, tk.seed)
    return (inst=g.inst, cond=cnd, p=p, method=tk.method, rep=tk.rep, curve=r.curve,
            ybest=r.ybest, ygrid=NaN, xbest=r.xbest, t_total=r.t_total, t_fit=r.t_fit, n_init=r.n_init)
end
end # @everywhere begin (kernel ablation)

if length(ARGS) >= 1 && ARGS[1] == "kernel_ablation"
    using Statistics, LinearAlgebra
    n_at        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    n_instances = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
    p_list      = length(ARGS) >= 4 ? parse.(Int, split(ARGS[4], ",")) : [1]
    budget      = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 300
    n_rep       = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
    methods     = length(ARGS) >= 7 ? Symbol.(split(ARGS[7], ",")) : collect(ABL_METHODS)
    n_grid      = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 60
    optimizer   = "zygote"
    ABL_CONDS   = ["cold_start", "gnn_warmstart"]      # p 固定时 tqa_init 与 cold_start 在纯 BO 下完全相同, 不重复
    for m in methods
        m in ABL_METHODS || error("未知方法 $(m); 可选: $(join(ABL_METHODS, ","))")
    end
    all(p -> 1 <= p <= 5, p_list) || error("p 列表只能取 1..5")
    budget >= 30 || error("预算至少 30")
    ref_m = methods[1]                                 # 配对比较的参照方法 = 列表第一个

    println("\n" * "="^70)
    println("核消融实验 (纯 BO, 无 NM 精修 / 无停滞重启)")
    println("n=$(n_at)  实例数=$(n_instances)  p=$(p_list)  预算=$(budget) 次演化  重复=$(n_rep)")
    println("方法: $(join(methods, ", "))   (配对参照: $(ref_m))")
    println("条件: $(join(ABL_CONDS, ", "))   p=1 标准答案: $(n_grid)×$(n_grid) 网格 + NM 精修")
    println("超参数拟合: $(GP_V2_HP_ITERS) 次迭代, 重启 2+$(GP_V2_EXTRA_RESTARTS) (v1 仍为 400 次/2 次重启)")
    :gsmcsm_v2hp in methods &&
        println("  gsmcsm_v2hp 的迭代次数: " * join(["p=$(p)(D=$(2p)): $(abl_hp_iters(2p))" for p in p_list], ", "))
    any(p -> p >= 2, p_list) &&
        println("  p≥2 为**不锁前缀**的联合 $(join(unique(2 .* p_list), "/")) 维 BO; 候选点 2000(80% 全局 + 20% 当前最优附近)")
    QAOA_BOX_SCALE != 1.0 && println("参数盒放大 ×$(QAOA_BOX_SCALE)")
    println("="^70)

    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" *
                 (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    isfile(prior_file) || error("找不到先验 $(prior_tag) —— 先跑一次 gnn_batch_qaoa_weighted 生成它")
    gnn_params = deserialize(prior_file)
    GRAPHS, INST_FPS, INST_SETFP, INST_FROM_FILE =
        obtain_test_instances(n_at, n_instances, gnn_params; verbose=false, save=true, prior_tag=prior_tag)

    abl_cfg = "abl_n$(n_at)_B$(budget)_G$(n_grid)_hp$(GP_V2_HP_ITERS)r$(GP_V2_EXTRA_RESTARTS)" *
              (QAOA_BOX_SCALE == 1.0 ? "" : "_box$(QAOA_BOX_SCALE)")
    abl_dir = joinpath(directoryBasis, "ckpt_" * abl_cfg)
    mkpath(abl_dir)
    check_ckpt_fingerprints(abl_dir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE)
    abl_fname(inst, cnd, p, m, rep) = "t_i$(inst)_$(cnd)_p$(p)_$(m)_r$(rep).jls"

    # 任务: 每个 (实例, 条件, p) 一个网格任务(仅 p=1) + 方法 × 重复
    ALL_T = NamedTuple[]
    for g in GRAPHS, cnd in ABL_CONDS, p in p_list
        p == 1 && push!(ALL_T, (g=g, cond=cnd, p=p, method=:grid, rep=0, budget=budget,
                                n_grid=n_grid, seed=0))
        for m in methods, rep in 1:n_rep
            sd = 100000 + 1000*g.inst + 100*rep + 10*p + (cnd == "gnn_warmstart" ? 1 : 0)
            push!(ALL_T, (g=g, cond=cnd, p=p, method=m, rep=rep, budget=budget, n_grid=n_grid, seed=sd))
        end
    end
    ABL_DONE = NamedTuple[]; ABL_TODO = NamedTuple[]
    for tk in ALL_T
        f = joinpath(abl_dir, abl_fname(tk.g.inst, tk.cond, tk.p, tk.method, tk.rep))
        if isfile(f)
            try
                push!(ABL_DONE, deserialize(f))
            catch
                push!(ABL_TODO, tk)
            end
        else
            push!(ABL_TODO, tk)
        end
    end
    @printf("任务: 共 %d 个, 已完成 %d, 本次计算 %d, worker=%d\n",
            length(ALL_T), length(ABL_DONE), length(ABL_TODO), nworkers())
    # 网格任务最慢, 先派发, 避免最后只剩一个大任务拖尾
    sort!(ABL_TODO, by = tk -> tk.method == :grid ? 0 : 1)
    t_abl = time()
    ABL_NEW = isempty(ABL_TODO) ? NamedTuple[] : pmap(ABL_TODO) do tk
        r = abl_task(tk)
        # 在 worker 上执行: 只用字符串拼路径(abl_dir 是 String), 不调主进程的函数
        try
            serialize(joinpath(abl_dir, "t_i$(tk.g.inst)_$(tk.cond)_p$(tk.p)_$(tk.method)_r$(tk.rep).jls"), r)
        catch e
            @warn "消融 checkpoint 写入失败(不影响本次计算)" exception=e
        end
        wlog(@sprintf("    [消融] inst%d %-13s p=%d %-10s r%d  best R_a=%.4f  [%.1f 分钟, 其中超参数拟合 %.0f%%]",
                      tk.g.inst, tk.cond, tk.p, String(tk.method), tk.rep, -r.ybest/tk.g.cmax,
                      r.t_total/60, 100*r.t_fit/max(r.t_total, 1e-9)), :always)
        r
    end
    ABL = vcat(ABL_DONE, ABL_NEW)
    @printf("\n消融计算完成, 墙钟 %.1f 分钟\n", (time()-t_abl)/60)

    # ── 可选: 并入基准(比如 hybrid)结果, 作为 p≥2 的参照并做"联合 vs 逐层"比较 ──
    base_name = get(ENV, "QAOA_ABL_BASELINE", "")
    BASE = Dict{Tuple{Int,String,Int},Float64}()        # (inst, cond, p) => R_a (多种子取 best)
    if !isempty(base_name)
        bdir = joinpath(directoryBasis, base_name)
        if isdir(bdir)
            check_ckpt_fingerprints(bdir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE, write=false)
            for g in GRAPHS, f in filter(x -> startswith(x, "task_i$(g.inst)_s") && endswith(x, ".jls"), readdir(bdir))
                r = try deserialize(joinpath(bdir, f)) catch; nothing end
                r === nothing && continue
                for cnd in ABL_CONDS, p in p_list
                    haskey(r.out[cnd], p) || continue
                    k = (g.inst, cnd, p)
                    BASE[k] = max(get(BASE, k, -Inf), r.out[cnd][p])
                end
            end
            println("并入基准 $(base_name): $(length(BASE)) 个 (实例,条件,p) 值")
        else
            println("⚠ QAOA_ABL_BASELINE=$(base_name) 目录不存在, 忽略")
        end
    end

    # ── 标准答案 ──
    cmax_of = Dict(g.inst => g.cmax for g in GRAPHS)
    REF = Dict{Tuple{Int,String,Int},Float64}()
    for g in GRAPHS, cnd in ABL_CONDS, p in p_list
        rs = [r for r in ABL if r.inst == g.inst && r.cond == cnd && r.p == p]
        isempty(rs) && continue
        vals = [-r.ybest/g.cmax for r in rs]
        haskey(BASE, (g.inst, cnd, p)) && push!(vals, BASE[(g.inst, cnd, p)])
        REF[(g.inst, cnd, p)] = maximum(vals)
    end
    # [注意] 顶层脚本作用域里不能在 for 循环中给全局变量累加(软作用域规则, 之前
    # nlines 就踩过), 这里用一次性求和表达式代替
    grid_val = Dict((r.inst, r.cond) => -r.ybest/cmax_of[r.inst] for r in ABL if r.method == :grid)
    n_beat_grid = count(r -> r.p == 1 && r.method != :grid && haskey(grid_val, (r.inst, r.cond)) &&
                             -r.ybest/cmax_of[r.inst] > grid_val[(r.inst, r.cond)] + 1e-6, ABL)
    1 in p_list && println("p=1 网格核查: 有 $(n_beat_grid) 次 BO 运行超过了网格+精修的值" *
                           (n_beat_grid == 0 ? " (网格确为最优, 标准答案可信)" :
                            " —— 网格分辨率可能不够, 标准答案已自动取所有方法中的最好值"))

    tol_s = 1e-3                                         # "找到最优"的判据: 差距 < 1e-3
    Bs = sort(unique(vcat(filter(b -> b <= budget, [25, 50, 100, 150, 200, 300, 400, 500]), budget)))
    regret_at(r, k) = REF[(r.inst, r.cond, r.p)] - (-r.curve[min(k, length(r.curve))] / cmax_of[r.inst])
    se_(v) = length(v) > 1 ? std(v)/sqrt(length(v)) : 0.0
    sign_p(w, l) = (n = w + l; n == 0 ? 1.0 :
                    min(1.0, 2*sum(Float64(binomial(BigInt(n), i)) for i in 0:min(w, l)) / 2.0^n))

    mprintln("\n" * "="^78)
    mprintln("【表A】与标准答案的差距 (regret, R_a 单位, 越小越好): 均值 ± 标准误")
    mprintln("      每格 = 实例 × 重复 次独立运行; '达标' = 差距 < $(tol_s); '中位评估数' = 达标者首次达标的演化次数")
    mprintln("="^78)
    SUMROWS = NamedTuple[]
    for p in p_list, cnd in ABL_CONDS
        mprintln("\n  p=$(p)  $(cnd)" * (p == 1 ? "  (标准答案 = 网格)" : "  (标准答案 = 所有方法" * (isempty(BASE) ? "" : " + 基准") * " 的最好值)"))
        mprintln("  " * rpad("方法", 11) * join([lpad("@$(b)", 17) for b in Bs]) * "   达标    中位评估数  未达标  分钟/次  拟合占比")
        for m in methods
            rs = [r for r in ABL if r.p == p && r.cond == cnd && r.method == m]
            isempty(rs) && continue
            cells = String[]
            for b in Bs
                v = [regret_at(r, b) for r in rs]
                push!(cells, lpad(@sprintf("%.4f±%.4f", mean(v), se_(v)), 17))
            end
            fin = [regret_at(r, budget) for r in rs]
            hit = [findfirst(k -> regret_at(r, k) < tol_s, 1:length(r.curve)) for r in rs]
            solved = [h for h in hit if h !== nothing]
            med = isempty(solved) ? NaN : median(solved)
            tmin = mean(r.t_total for r in rs)/60
            fshare = mean(r.t_fit/max(r.t_total, 1e-9) for r in rs)
            mprintln("  " * rpad(String(m), 11) * join(cells) *
                     @sprintf("   %3d/%-3d %9s   %4d   %6.2f   %5.0f%%",
                              length(solved), length(rs), isnan(med) ? "-" : string(round(Int, med)),
                              length(rs) - length(solved), tmin, 100*fshare))
            push!(SUMROWS, (p=p, cond=cnd, method=m, n=length(rs),
                            regrets=[mean([regret_at(r, b) for r in rs]) for b in Bs],
                            solved=length(solved), median_evals=med, minutes=tmin, fit_share=fshare))
        end
    end

    mprintln("\n" * "="^78)
    mprintln("【表B】配对比较: $(ref_m) vs 其它方法 (同实例/同条件/同重复 = 同一初始设计)")
    mprintln("      Δ = 参照方法的最好 R_a − 对手的最好 R_a (>0 表示 $(ref_m) 更好); |Δ|<1e-4 记持平")
    mprintln("="^78)
    for p in p_list, cnd in ABL_CONDS, b in unique([min(100, budget), budget])
        for m in methods
            m == ref_m && continue
            d = Float64[]
            for r in ABL
                (r.p == p && r.cond == cnd && r.method == ref_m) || continue
                o = findfirst(x -> x.p == p && x.cond == cnd && x.method == m && x.inst == r.inst && x.rep == r.rep, ABL)
                o === nothing && continue
                push!(d, regret_at(ABL[o], b) - regret_at(r, b))
            end
            isempty(d) && continue
            w = count(>(1e-4), d); l = count(<(-1e-4), d); t = length(d) - w - l
            @printf("  p=%d %-13s @%-4d  vs %-10s  Δ=%+.5f ± %.5f   赢/平/输 %3d/%3d/%3d   p_sign=%.3g\n",
                    p, cnd, b, String(m), mean(d), se_(d), w, t, l, sign_p(w, l))
        end
    end

    if !isempty(BASE)
        mprintln("\n" * "="^78)
        mprintln("【表C】联合 BO(不锁前缀, 纯 BO) vs 基准 $(base_name), 每实例取各方法 $(n_rep) 次重复中的最好值")
        mprintln("      >0 表示联合 BO 更好。注意: 基准含 NM 精修与轮扫, 联合 BO 这里是纯 BO —— 这是对联合 BO 不利的比较,")
        mprintln("      若联合 BO 仍能在多数实例上胜出, 说明放开前缀确有收益。")
        mprintln("="^78)
        for p in p_list, cnd in ABL_CONDS, m in methods
            d = Float64[]
            for g in GRAPHS
                haskey(BASE, (g.inst, cnd, p)) || continue
                rs = [r for r in ABL if r.inst == g.inst && r.cond == cnd && r.p == p && r.method == m]
                isempty(rs) && continue
                push!(d, maximum(-r.ybest/g.cmax for r in rs) - BASE[(g.inst, cnd, p)])
            end
            isempty(d) && continue
            w = count(>(1e-4), d); l = count(<(-1e-4), d)
            @printf("  p=%d %-13s %-10s  Δ=%+.5f ± %.5f   赢/平/输 %2d/%2d/%2d   p_sign=%.3g\n",
                    p, cnd, String(m), mean(d), se_(d), w, length(d)-w-l, l, sign_p(w, l))
        end
    end

    # ── CSV ──
    tagc = abl_cfg * (INST_FROM_FILE ? "_I$(INST_SETFP)" : "")
    fc = joinpath(directoryBasis, "ablation_curves_$(tagc).csv")
    open(fc, "w") do io
        println(io, "instance,condition,p,method,rep,evals,best_ar,regret")
        for r in ABL
            r.method == :grid && continue
            for k in unique(vcat(collect(5:5:length(r.curve)), length(r.curve)))
                @printf(io, "%d,%s,%d,%s,%d,%d,%.6f,%.6f\n", r.inst, r.cond, r.p, String(r.method), r.rep,
                        k, -r.curve[k]/cmax_of[r.inst], regret_at(r, k))
            end
        end
    end
    fs = joinpath(directoryBasis, "ablation_summary_$(tagc).csv")
    open(fs, "w") do io
        println(io, "p,condition,method,runs," * join(["regret_at_$(b)" for b in Bs], ",") *
                    ",solved,median_evals_to_solve,minutes_per_run,hp_fit_share")
        for s in SUMROWS
            println(io, join((s.p, s.cond, s.method, s.n, (@sprintf("%.6f", x) for x in s.regrets)...,
                              s.solved, isnan(s.median_evals) ? "" : string(s.median_evals),
                              @sprintf("%.3f", s.minutes), @sprintf("%.3f", s.fit_share)), ","))
        end
    end
    mprintln("\n✓ 收敛曲线CSV → $(basename(fc))   (画 regret–演化次数 曲线用)")
    mprintln("✓ 汇总CSV     → $(basename(fs))")
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── transfer_bo: 跨实例迁移 / 多任务 GP 的贝叶斯优化 ─────────────────────────
#
# 【理论基础】
#  (1) 参数集中 (parameter concentration): 同一分布里抽出的 MaxCut 实例, QAOA 最优
#      参数聚在一起, 所以一个实例上优化出来的参数可以直接用到另一个实例上。
#      · Shaydulin et al., ACM TQC 4, 3 (2023) [arXiv:2201.11785]: 加权 MaxCut 的
#        参数迁移。关键是一条重标度规则(其 Eq. 9-10):
#            β_w = β_S,   γ_w = γ_S · arctan(1/√(d−1)) / w̄,   w̄ = (1/|E|) Σ|w_ij|
#        即 γ·w̄ 是跨实例不变量。直接迁移只损失约 2.0 个百分点, 再做一次局部优化,
#        96% 的情况能追平穷举。
#      · Montañez-Barrera, Willsch, Michielsen (arXiv:2402.05549): 先把代价哈密顿量
#        除以最大系数 max|w| 再直接迁移(γ·max|w| 不变), 能跨实例、跨尺寸迁移。
#  (2) 映射到我们的模拟平台: 代价层(Ω=0, 时长 γ)里 ZZ 系数就是 J_ij = C6/(4 r_ij^6),
#      与 g.weights 同单位(target3d 嵌入正是按 w = C6/(4r^6) 反推距离)。所有实例都是
#      3-正则, arctan(1/√2) 是常数, 抵消。所以:
#          γ_t = γ_s · κ_s / κ_t,   β_t = β_s,
#      κ = 边上 J_ij 的均值(QAOA_TL_RESCALE=mean, Shaydulin; 默认) 或最大值(=max,
#      Montañez-Barrera) 或不重标度(=none, 消融对照)。混合器 Ω 固定 ⇒ β 不变。
#  (3) 从"参数集中"到"能量曲面集中": 把每一层的目标写成**层增益**
#          g(u) = −[R_a(前缀⊕u) − R_a(前缀⊕(0,0))],    u = (γ_p, β_p)
#      (0,0) 是恒等层, 它的值就是上一层最优值, 精确已知、不花演化。这样减掉了实例
#      之间的整体水平差(有的实例 0.63, 有的 0.74), 剩下的增益曲面在重标度后应当
#      相似。于是对新实例 t:  g_t(u) ≈ m_S(u) + δ_t(u),  m_S 是源实例的群体曲面,
#      δ_t 是实例特有的小偏差 —— 这正是迁移 GP 的模型假设。
#
# 【方法】(每层都是锁前缀的二维搜索, 与生产 hybrid 的逐层结构一致; 每层预算 B 次演化)
#   matern52 : 只用目标数据的 Matérn-5/2 BO(= 核消融里胜出的那个, 无迁移基线)
#   tl_nm    : Shaydulin 协议 —— 从迁移中位数参数出发做局部 NM(无 GP)
#   tl_init  : Matérn BO, 但初始设计的前 5 个点换成迁移点(中位数 + 4 个 KDE 高密度
#              源最优, 对应 Shaydulin 的 KDE 元分布采样), 其余仍是 Halton
#   tl_mean  : 堆叠/残差迁移 GP: 先在源数据上拟合群体 GP m_S, 目标 GP 的先验均值
#              = m_S(u), 残差 δ_t 的超参数由边际似然拟合(残差长度尺度的先验取源 GP 的
#              长度尺度 —— 超参数也迁移)。源 GP 的后验不确定性一并计入。
#   tl_mtgp  : 多任务 GP(ICM 共区域化核, Bonilla et al. 2008 / Swersky et al. 2013):
#                k((u,i),(u',j)) = σ²·B_ij·k52(u,u'),  B = [1 ρa; ρa a²]
#              源(合并为一个任务)与目标的相关系数 ρ 由数据学出 —— 如果目标与源不
#              相似, ρ→0, 自动退化成单任务 BO(不会被坏的源带偏)。源实例之间的差异
#              用一个可学习的源噪声 σ_S² 吸收。
#   tl_mean / tl_mtgp 的初始设计只有 5 个迁移点(先验已含源信息, 不再需要空间填充)。
#
# 【协议】留一法(LOO): 目标实例 t 的源只来自其它实例, 绝不含 t 自己。
#   阶段 0: p=1 网格标准答案(能复用核消融目录里已算好的就直接读)。
#   阶段 1: 源数据库 —— 每个 (实例, 条件) 跑一次逐层链: Matérn BO(B_src 次)+
#           二维 NM 精修(40 次), 保存每层所有评估点和最优点。对应"以前跑过的实例"。
#   阶段 2: 目标运行 —— 每个 (实例, 条件, 方法, 重复) 跑一次 p=1..p_max 逐层链,
#           每层只给 B 次演化(默认 30, 生产 hybrid 每层约 300 次 + 轮扫 + 精修)。
#   源实例个数 K 可调(默认 = 其它全部实例), 用来画"迁移效果 vs 源数目"。
#
# 运行方式:
#   julia r3g_sge_integrated.jl transfer_bo <n_atoms> <n_instances> <p_max> <每层预算B>
#                                           <重复数> [方法列表] [源个数K] [源预算B_src]
# 例如:
#   ... transfer_bo 12 20 5 30 2
#   ... transfer_bo 12 20 3 20 2 matern52,tl_nm,tl_mtgp 5
# 环境变量: QAOA_TL_RESCALE=mean|max|none (默认 mean); QAOA_INSTANCES; QAOA_ABL_BASELINE
# ══════════════════════════════════════════════════════════════════════════
@everywhere begin
const TL_METHODS    = (:matern52, :tl_nm, :tl_init, :tl_mean, :tl_mtgp, :tl_hyb, :tl_full, :m52_hyb,
                       :tl_port, :tl_portp, :tl_portg, :tl_hybx, :tl_hyb3, :tl_hybs2, :tl_hybs3,
                       :tl_ann, :tl_annx, :tl_mc, :tl_mcs, :tl_mcm, :tl_mca, :tl_mcx)
# 走 "新层 BO → 多起点 → 轮扫 → 全维精修" 流水线的方法(只差新层的开局点 / 源数据库 / 预算)
#   tl_hyb3: 与 tl_hyb 完全相同, 但所有预算(新层 BO、多起点、轮扫、精修)都 ×M —— tl_mc 的等预算对照
#   tl_hybs2 / tl_hybs3: 与 tl_hyb 完全相同, 只换随机种子 —— 构成"只靠种子的多样性"对照 tl_mcs
#   tl_ann: "退火分支"链(冷启动和 GNN 预热都可用): 第 1 层的 γ 始终限制在 [0, QAOA_TL_ANN_GMAX](默认 1 μs),
#           其余与 tl_hyb 相同; 源 = 其它实例的"退火分支"源链(同一条件, 第 1 层也限制 γ ≤ GMAX)。
#           GNN 预热的最优开局也有两类(γ≈0.4 / γ≈2.5), 同样的约束检验小 γ 分支在深层是否更好。
#           动机: Ωt=15.7 冷启动有两类最优路径 —— 小 γ 平滑退火分支 p=1→5 增益 +0.170, 大 γ 分支只有 +0.087,
#           而逐层贪心在 p=1 常常选中大 γ(p=1 更好)。这条链检验"走对分支能否在深层持续领先"。
#   tl_annx: "全层退火"链 —— 与 tl_ann 相同, 但**每一层**的 γ 都限制在 [0, GMAX](BO 新层/多起点/轮扫/精修/INTERP 全程)。
#           动机: tl_ann 只限制第 1 层, 6 个停滞实例(i2/4/5/8/14/16)在第 2 层就跳到 γ≈1.4–1.8 的非绝热盆地,
#           之后的层退化成 β≈0 的纯相位层; 正常实例(如 i3)整条曲线 γ ≤ 0.7, 不受这个约束影响。
#           源 = 同样全层受限的退火源链(srcannx_*), 保证分支匹配。
const TL_HYB_LIKE   = (:tl_hyb, :m52_hyb, :tl_port, :tl_portp, :tl_portg, :tl_hybx, :tl_hyb3, :tl_hybs2, :tl_hybs3, :tl_ann, :tl_annx)
const TL_ANN_M      = (:tl_ann, :tl_annx)
# 虚拟(合成)方法: tl_mc = 开局不同的 3 条迁移链; tl_mcs = tl_hyb 的 3 个种子(开局相同);
#                 tl_mcm = tl_hyb + tl_port + m52_hyb(第 3 条链不用迁移, 看能否补上稀有峰)
#                 tl_mca = tl_mc 的 3 条链 + tl_ann
const TL_VIRT_M     = (:tl_mc, :tl_mcs, :tl_mcm, :tl_mca, :tl_mcx)   # tl_mcx = tl_mca 的链 + tl_annx
# tl_mc(多链迁移)不是独立运行的方法: 主程序把 M 条开局不同的链(默认 tl_hyb / tl_port / tl_portp,
#   同一实例/条件/重复)组合起来, 每个深度取 R_a 最好的那条; 演化数 = 各链之和。
#   取优只用每条链自己测得的 R_a(优化时本来就要算的量), 不用任何标准答案。
# 覆盖式开局组合: tl_port 源等权; tl_portp 按物理/嵌入特征加权; tl_portg 按 GNN 输出特征加权
const TL_PORT_METHODS = (:tl_port, :tl_portp, :tl_portg)
const TL_RESCALE    = lowercase(get(ENV, "QAOA_TL_RESCALE", "mean"))
const TL_N_TRANSFER = 5          # 迁移初始点: 中位数 + 4 个 KDE 高密度源最优
const TL_SRC_CAP    = 300        # 源 GP / 多任务 GP 里源数据点上限(控制 O(N³))
const TL_SRC_TOPK   = 8          # 每个源实例必保留的"好点"个数(互相分开)
const TL_SRC_POLISH = 40         # 源数据库每层的二维 NM 精修演化数
const TL_REFIT      = 5          # 所有 GP 方法: 每 5 次新评估重拟合一次超参数
const TL_XI0        = 0.005      # EI 的 ξ(R_a 单位; 约等于核消融里原始单位 0.05 / C_max)
# 预热正则化 ε (Egger et al., Quantum 5, 479 (2021)): GNN 概率先夹到 [ε, 1−ε] 再制备初态。
#   ε=0(默认, 与以往所有结果一致): GNN 很自信(p≈0/1)的原子, 初态正好是定制 mixer 的本征态,
#   mixer 转不动它, 代价层又是对角的 ⇒ 这个原子被永久冻结; GNN 判错的自信原子任何角度都救不回。
#   ε>0: 给每个原子留一点"可翻转"的余地, 代价是初态里 GNN 信息打了折扣。
const TL_WS_EPS     = parse(Float64, get(ENV, "QAOA_WS_EPS", "0.0"))
tl_ws_p(g) = TL_WS_EPS > 0 ? clamp.(g.best_p_gnn, TL_WS_EPS, 1.0 - TL_WS_EPS) : g.best_p_gnn
const TL_WS_MIXSIGN = parse(Float64, get(ENV, "QAOA_WS_MIXSIGN", "1"))   # −1: Egger 精确版定制 mixer(见 loss_fn_warmstart_weighted)
TL_WS_MIXSIGN in (1.0, -1.0) || error("QAOA_WS_MIXSIGN 只能是 1 或 -1")
# [B 方案] gnn_bias 条件的偏置场参数: λ_k = λ0·r^(k−1); prep=1 ⇒ 初态取第 1 层混合层的本征态
const TL_BIAS_L0   = parse(Float64, get(ENV, "QAOA_BIAS_L0", "1.0"))
const TL_BIAS_R    = parse(Float64, get(ENV, "QAOA_BIAS_R", "0.7"))
const TL_BIAS_PREP = get(ENV, "QAOA_BIAS_PREP", "1") == "1"
# geo: B1(λ_k = λ0·r^(k−1)); lin: λ_k = λ0·(1 − (k−1)/p); interp: B2(Egger 精确 mixer → 标准 mixer 线性过渡, 无自由参数)
const TL_BIAS_MODE = Symbol(get(ENV, "QAOA_BIAS_MODE", "geo"))
TL_BIAS_MODE in (:geo, :lin, :interp, :egeo) || error("QAOA_BIAS_MODE 只能是 geo / lin / interp / egeo")
tl_coff(c) = c == "gnn_warmstart" ? 1 : (c == "gnn_bias" ? 2 : 0)   # 不同条件的随机种子偏移
# 混合层随总深度 p 变化的条件(gnn_bias 的 lin / interp): 加一层后前缀的哈密顿量也变了,
# "新层取 (0,0) 时 = 上一深度的最优值"不再成立 ⇒ 原点必须真实评估, 不能用 yprev 免费填入。
tl_depth_dep(c) = c == "gnn_bias" && TL_BIAS_MODE in (:lin, :interp)
struct TLBudgetHit <: Exception end

"""实例的耦合尺度 κ。J_ij = C6/(4 r_ij^6) 取自真实原子位置(哈密顿量实际看到的系数)。"""
function tl_kappa(g, mode::AbstractString)
    mode == "none" && return 1.0
    C6 = 2π*862690
    J = Float64[]
    for e in g.edges
        a = g.atoms[e[1]]; b = g.atoms[e[2]]
        r2 = (a[1]-b[1])^2 + (a[2]-b[2])^2 + (a[3]-b[3])^2
        push!(J, C6 / (4 * r2^3))
    end
    mode == "mean" && return mean(J)
    mode == "max"  && return maximum(J)
    error("QAOA_TL_RESCALE 只能是 mean / max / none, 收到 $(mode)")
end
tl_kappa_of(rec, mode::AbstractString) = mode == "mean" ? rec.kmean : (mode == "max" ? rec.kmax : 1.0)

"""返回 x ↦ −R_a(x)(越小越好)。失败点记为 R_a = 0(不会把 GP 拉坏)。"""
function tl_make_evf(g, cnd::String)
    return x -> begin
        hi = repeat([BOX_HI[1], BOX_HI[2]], length(x) ÷ 2)
        xq = clamp.(x, 0.0, hi)
        lv = try
            v = cnd == "gnn_warmstart" ?
                loss_fn_warmstart_weighted(xq, tl_ws_p(g), g.atoms, g.edges, g.weights, g.n_atoms; mixsign=TL_WS_MIXSIGN) :
                cnd == "gnn_bias" ?
                loss_fn_bias_weighted(xq, tl_ws_p(g), g.atoms, g.edges, g.weights, g.n_atoms;
                                      λ0=TL_BIAS_L0, r=TL_BIAS_R, prep=TL_BIAS_PREP, mode=TL_BIAS_MODE) :
                loss_fn_weighted(xq, g.atoms, g.edges, g.weights, g.n_atoms)
            isfinite(v) ? clamp(v, -(g.total_w + 1.0), 0.0) : 0.0
        catch
            0.0
        end
        lv / g.cmax
    end
end

"""ARD Matérn-5/2 相关函数(无幅度), 输入已按量程归一化。"""
function tl_m52(A::Matrix{Float64}, B::Matrix{Float64}, ℓ::AbstractVector)
    r2 = zeros(size(A, 2), size(B, 2))
    for d in 1:size(A, 1)
        Δ = A[d:d, :]' .- B[d:d, :]
        r2 .+= (Δ ./ ℓ[d]).^2
    end
    r = sqrt.(r2)
    return (1 .+ sqrt(5) .* r .+ (5/3) .* r2) .* exp.(-sqrt(5) .* r)
end

function tl_minimize(nll, lo::Vector{Float64}, hi::Vector{Float64},
                     starts::Vector{Vector{Float64}}; iters::Int=600)
    bθ = clamp.(starts[1], lo, hi); bv = Inf
    for θ0 in starts
        θc = clamp.(θ0, lo .+ 1e-6, hi .- 1e-6)
        r = try
            optimize(nll, lo, hi, θc, Fminbox(NelderMead()), Optim.Options(iterations=iters, outer_iterations=3))
        catch
            nothing
        end
        r === nothing && continue
        Optim.minimum(r) < bv && (bv = Optim.minimum(r); bθ = Optim.minimizer(r))
    end
    return bθ
end

# ── 源群体 GP(带噪声: 源实例之间的差异当作噪声吸收) ──
struct TLSrc
    Xn::Matrix{Float64}; L::LowerTriangular{Float64,Matrix{Float64}}; α::Vector{Float64}
    s2::Float64; ℓ::Vector{Float64}; m0::Float64; sc::Float64; sn2::Float64
end
function tl_ystd(y::Vector{Float64})
    m0 = mean(y); sc = length(y) > 1 ? std(y) : 1.0
    (isfinite(sc) && sc > 1e-12) || (sc = 1.0)
    return m0, sc
end
function tl_nll_single(Xn::Matrix{Float64}, yz::Vector{Float64}, θ::Vector{Float64})
    try
        K = exp(θ[1]) .* tl_m52(Xn, Xn, exp.(θ[2:3])); K[diagind(K)] .+= exp(θ[4]) + 1e-8
        L = cholesky(Symmetric(K)).L; α = L' \ (L \ yz)
        v = 0.5*dot(yz, α) + sum(log.(diag(L))) + 0.5*sum(abs2, θ[1:3]) + 0.5*((θ[4] + 4)/2)^2
        return isfinite(v) ? v : 1e10
    catch
        return 1e10
    end
end
function tl_fit_build_src(U::Matrix{Float64}, yg::Vector{Float64}, xs::Vector{Float64})
    Xn = U ./ xs; m0, sc = tl_ystd(yg); yz = (yg .- m0) ./ sc
    lo = [-6.0, -6.0, -6.0, -12.0]; hi = [6.0, 6.0, 6.0, 1.0]
    starts = [[0.0, log(0.2), log(0.2), -4.0], [0.0, log(0.5), log(0.5), -2.0],
              [0.0, log(0.1), log(0.3), -6.0]]
    θ = tl_minimize(θ -> tl_nll_single(Xn, yz, θ), lo, hi, starts; iters=600)
    K = exp(θ[1]) .* tl_m52(Xn, Xn, exp.(θ[2:3])); K[diagind(K)] .+= exp(θ[4]) + 1e-8
    L, _ = safe_chol(K, 1e-10)
    return TLSrc(Xn, L, L' \ (L \ yz), exp(θ[1]), exp.(θ[2:3]), m0, sc, exp(θ[4]))
end
"""源 GP 的潜函数后验(原始增益单位)。返回 (μ, Σ) —— Σ 为完整协方差。"""
function tl_src_post(m::TLSrc, Cn::Matrix{Float64})
    Kx = m.s2 .* tl_m52(m.Xn, Cn, m.ℓ); v = m.L \ Kx
    μ = m.m0 .+ m.sc .* (Kx' * m.α)
    Σ = m.sc^2 .* (m.s2 .* tl_m52(Cn, Cn, m.ℓ) .- v' * v)
    return μ, (Σ .+ Σ') ./ 2
end

# ── tl_mean: 目标 = m_S + δ_t (残差/堆叠迁移 GP, Golovin et al. 2017 的 stacking 思路) ──
#   θ = [log s2_δ, log ℓ1, log ℓ2, b]; b 固定为 1(只为接口与诊断保留)。
#   [设计记录] 最初让缩放 b 也由边际似然拟合。Python 原型发现: 目标数据后期会聚在
#   自身最优附近(相距 ~0.005 量程、取值几乎相同), 而源均值在那里有残余斜率(源最优
#   略有偏移)。在近乎奇异的协方差下, 广义最小二乘被这些"几乎重复点之间的差"主导,
#   b 被压到 ≈0.05(普通最小二乘给出 b≈1.03) —— 迁移信息被白白丢掉。固定 b=1、让
#   残差 δ 吸收幅度差异, 结果同样好且没有这个病态; 需要可学习幅度/相关性的场合交给
#   tl_mtgp(它的 ρ, a 有同样的作用但不受这个问题影响, 原型里 ρ≈0.9)。
function tl_fit_mean(src::TLSrc, Un::Matrix{Float64}, yg::Vector{Float64}; θ_prev=nothing)
    θp = [log(0.1), log(src.ℓ[1]), log(src.ℓ[2]), 1.0]          # 先验中心(超参数迁移)
    size(Un, 2) < 3 && return θp
    μS, ΣS = tl_src_post(src, Un)
    rz = (yg .- μS) ./ src.sc; Σz = ΣS ./ src.sc^2
    lo = [-8.0, -6.0, -6.0]; hi = [2.0, 6.0, 6.0]
    function nll(t3)
        try
            K = Σz .+ exp(t3[1]) .* tl_m52(Un, Un, exp.(t3[2:3])); K = (K .+ K') ./ 2
            K[diagind(K)] .+= 1e-6
            L = cholesky(Symmetric(K)).L; α = L' \ (L \ rz)
            v = 0.5*dot(rz, α) + sum(log.(diag(L))) + 0.5*(t3[1] - θp[1])^2 +
                0.5*sum(abs2, (t3[2:3] .- θp[2:3]) ./ 0.5)
            return isfinite(v) ? v : 1e10
        catch
            return 1e10
        end
    end
    starts = θ_prev === nothing ? [θp[1:3]] : [Vector{Float64}(θ_prev[1:3]), θp[1:3]]
    return vcat(tl_minimize(nll, lo, hi, starts; iters=400), 1.0)
end
function tl_mean_post(src::TLSrc, θ::Vector{Float64}, Un::Matrix{Float64}, yg::Vector{Float64},
                      Cn::Matrix{Float64})
    b = θ[4]; s2r = exp(θ[1]); ℓr = exp.(θ[2:3]); nt = size(Un, 2)
    μZ, ΣZ = tl_src_post(src, hcat(Un, Cn))
    μZ = μZ ./ src.sc; ΣZ = ΣZ ./ src.sc^2
    ic = (nt+1):(nt+size(Cn, 2))
    Kcc = b^2 .* ΣZ[ic, ic] .+ s2r .* tl_m52(Cn, Cn, ℓr)
    if nt == 0
        return src.sc .* b .* μZ[ic], src.sc^2 .* Kcc
    end
    Ktt = b^2 .* ΣZ[1:nt, 1:nt] .+ s2r .* tl_m52(Un, Un, ℓr); Ktt = (Ktt .+ Ktt') ./ 2
    Ktt[diagind(Ktt)] .+= 1e-6
    Kct = b^2 .* ΣZ[ic, 1:nt] .+ s2r .* tl_m52(Cn, Un, ℓr)
    L, _ = safe_chol(Ktt, 1e-10)
    α = L' \ (L \ (yg ./ src.sc .- b .* μZ[1:nt])); A = L \ Kct'
    μ = b .* μZ[ic] .+ Kct * α
    Σ = Kcc .- A' * A
    return src.sc .* μ, src.sc^2 .* ((Σ .+ Σ') ./ 2)
end

# ── tl_mtgp: ICM 双任务 GP ──  θ = [log σ², log ℓ1, log ℓ2, logit ρ, log a, log σ_S²]
function tl_mtgp_cov(θ::Vector{Float64}, A::Matrix{Float64}, ta::Vector{Int},
                     B::Matrix{Float64}, tb::Vector{Int})
    s2 = exp(θ[1]); ρ = 1/(1 + exp(-θ[4])); a = exp(θ[5])
    K = s2 .* tl_m52(A, B, exp.(θ[2:3]))
    @inbounds for j in 1:size(B, 2), i in 1:size(A, 2)
        K[i, j] *= ta[i] == tb[j] ? (ta[i] == 1 ? a^2 : 1.0) : ρ*a
    end
    return K
end
function tl_mtgp_data(src, Un::Matrix{Float64}, yg::Vector{Float64}, xs::Vector{Float64})
    XS = src.U ./ xs; m0, sc = tl_ystd(src.yg)
    X = hcat(XS, Un); z = (vcat(src.yg, yg) .- m0) ./ sc
    tk = vcat(zeros(Int, size(XS, 2)), ones(Int, size(Un, 2)))
    return X, z, tk, m0, sc
end
function tl_mtgp_addnoise!(K::Matrix{Float64}, tk::Vector{Int}, θ::Vector{Float64})
    for i in eachindex(tk)
        K[i, i] += tk[i] == 0 ? exp(θ[6]) + 1e-8 : 1e-6
    end
    return K
end
function tl_fit_mtgp(src, Un::Matrix{Float64}, yg::Vector{Float64}, xs::Vector{Float64}; θ_prev=nothing)
    X, z, tk, _, _ = tl_mtgp_data(src, Un, yg, xs)
    θp = [0.0, log(0.2), log(0.2), 1.5, 0.0, -4.0]
    lo = [-6.0, -6.0, -6.0, -4.0, -3.0, -12.0]; hi = [6.0, 6.0, 6.0, 6.0, 2.0, 1.0]
    function nll(θ)
        try
            K = tl_mtgp_addnoise!(tl_mtgp_cov(θ, X, tk, X, tk), tk, θ)
            L = cholesky(Symmetric(K)).L; α = L' \ (L \ z)
            v = 0.5*dot(z, α) + sum(log.(diag(L))) + 0.5*sum(abs2, θ[1:3]) +
                0.5*((θ[4] - 1.5)/1.5)^2 + 0.5*(θ[5]/0.5)^2 + 0.5*((θ[6] + 4)/2)^2
            return isfinite(v) ? v : 1e10
        catch
            return 1e10
        end
    end
    if θ_prev === nothing
        starts = [θp, [0.0, log(0.5), log(0.5), 0.5, 0.0, -2.0], [0.0, log(0.1), log(0.3), 3.0, 0.0, -6.0]]
        return tl_minimize(nll, lo, hi, starts; iters=600)
    end
    return tl_minimize(nll, lo, hi, [Vector{Float64}(θ_prev), θp]; iters=300)
end
function tl_mtgp_post(src, θ::Vector{Float64}, Un::Matrix{Float64}, yg::Vector{Float64},
                      xs::Vector{Float64}, Cn::Matrix{Float64})
    X, z, tk, m0, sc = tl_mtgp_data(src, Un, yg, xs)
    L, _ = safe_chol(tl_mtgp_addnoise!(tl_mtgp_cov(θ, X, tk, X, tk), tk, θ), 1e-10)
    tc = ones(Int, size(Cn, 2))
    Kc = tl_mtgp_cov(θ, X, tk, Cn, tc); v = L \ Kc; α = L' \ (L \ z)
    μ = m0 .+ sc .* (Kc' * α)
    Σ = sc^2 .* (tl_mtgp_cov(θ, Cn, tc, Cn, tc) .- v' * v)
    return μ, (Σ .+ Σ') ./ 2
end

# ── 采集 ──
function tl_ts_idx(μ::Vector{Float64}, Σ::Matrix{Float64}; ns::Int=10)
    Lc, _ = safe_chol(Σ, 1e-10 * max(maximum(diag(Σ)), 1e-12))
    bi = 1; bs = Inf
    for _ in 1:ns
        s = μ .+ Lc * randn(length(μ)); k = argmin(s)
        s[k] < bs && (bs = s[k]; bi = k)
    end
    return bi
end
"""与核消融相同的日程: t<0.30 TS(调用方处理) → EI → LCB。"""
tl_pick_idx(μ, σ, yb, t) = t < 0.68 ?
    argmax(ei_acq(μ, σ, yb; ξ=TL_XI0*exp(-3.0*(t-0.30)/0.38))) :
    argmin(lcb_acq(μ, σ; β=max(exp(-2.5*(t-0.68)/0.32), 0.05)))

"""一层锁前缀的二维搜索。f(u) = −R_a(前缀⊕u)。y0_known = 上一层最优(p≥2 时原点免费);
   p=1 时传 nothing, 原点要花 1 次演化。src = tl_source_pool 的结果(matern52 可为 nothing)。"""
function tl_layer(f, method::Symbol, B::Int, seed::Int, y0_known, src; polish::Int=0, ghi::Float64=BOX_HI[1])
    lo = [0.0, 0.0]; hi = [ghi, BOX_HI[2]]; xs = hi .- lo        # ghi < BOX_HI[1]: 本层 γ 的盒子被收窄(tl_ann 的第 1 层)
    Ucols = Vector{Vector{Float64}}(); Y = Float64[]; curve = Float64[]
    best = Ref(Inf); bu = Ref([0.0, 0.0]); nev = Ref(0); cap = Ref(B)
    ev! = function (u)
        nev[] >= cap[] && throw(TLBudgetHit())
        uc = clamp.(u, lo, hi); v = f(uc); nev[] += 1
        push!(Ucols, uc); push!(Y, v)
        v < best[] && (best[] = v; bu[] = copy(uc))
        push!(curve, best[])
        return v
    end
    dg = Dict{Symbol,Float64}()
    if y0_known === nothing
        y0 = ev!([0.0, 0.0])
    else
        y0 = Float64(y0_known)
        push!(Ucols, [0.0, 0.0]); push!(Y, y0); best[] = y0; bu[] = [0.0, 0.0]
    end
    Random.seed!(seed)
    t0 = time(); tfit = 0.0
    try
        n_init = min(24, max(12, B ÷ 8))
        design = Vector{Vector{Float64}}()
        method in (:tl_init, :tl_mean, :tl_mtgp) && append!(design, src.pts)
        if method in (:matern52, :tl_init)
            for h in halton_samples(n_init - 1, lo, hi; skip=7)
                length(design) >= n_init - 1 && break
                push!(design, clamp.(h .+ randn(2) .* 0.05 .* xs, lo, hi))
            end
        end
        for u in design
            ev!(u)
        end
        if method == :tl_nm
            z0 = clamp.(src.pts[1] ./ xs, 0.0, 1.0)
            optimize(z -> ev!(clamp.(z, 0.0, 1.0) .* xs), z0,
                     NelderMead(initial_simplex=Optim.AffineSimplexer(0.05, 0.0)),
                     Optim.Options(iterations=100000, g_tol=1e-12))
        elseif method in (:matern52, :tl_init, :tl_mean, :tl_mtgp)
            n_iter = B - nev[]; it = 0; θ = nothing; last_fit = -10^9; srcgp = nothing
            if method == :tl_mean
                tf = time(); srcgp = tl_fit_build_src(src.U, src.yg, xs); tfit += time() - tf
                dg[:src_ell_g] = srcgp.ℓ[1]; dg[:src_ell_b] = srcgp.ℓ[2]
            end
            while nev[] < B
                it += 1; t = it / max(n_iter, 1)
                U = reduce(hcat, Ucols); yg = Y .- y0; yb = minimum(yg)
                needfit = θ === nothing || nev[] - last_fit >= TL_REFIT
                # 候选点: 300 全局均匀 + 100 当前最优附近(两档尺度)。所有 GP 方法相同。
                # 纯均匀 400 点在二维的分辨率只有约 量程/20, 小预算下会卡在 ~1e-3 的差距
                # 上不去(Python 原型实测), 局部候选让 GP 也能像 NM 一样做精细收尾。
                bz = bu[]
                C = hcat(lo .+ rand(2, 300) .* xs,
                         clamp.(bz .+ randn(2, 50) .* 0.02 .* xs, lo, hi),
                         clamp.(bz .+ randn(2, 50) .* 0.005 .* xs, lo, hi))
                local xn
                if method in (:matern52, :tl_init)
                    if needfit
                        tf = time(); θ = abl_fit(:matern52, U, yg, 2, xs; θ_init=θ)
                        tfit += time() - tf; last_fit = nev[]
                    end
                    ga = abl_build(:matern52, U, yg, θ, 2, xs)
                    if t < 0.30
                        xn = C[:, abl_ts(ga, C; ns=10)]
                    else
                        μ, σ = abl_pred(ga, C); xn = C[:, tl_pick_idx(μ, σ, yb, t)]
                    end
                else
                    Un = U ./ xs; Cn = C ./ xs
                    if needfit
                        tf = time()
                        θ = method == :tl_mean ? tl_fit_mean(srcgp, Un, yg; θ_prev=θ) :
                                                 tl_fit_mtgp(src, Un, yg, xs; θ_prev=θ)
                        tfit += time() - tf; last_fit = nev[]
                    end
                    μ, Σ = method == :tl_mean ? tl_mean_post(srcgp, θ, Un, yg, Cn) :
                                                tl_mtgp_post(src, θ, Un, yg, xs, Cn)
                    xn = t < 0.30 ? C[:, tl_ts_idx(μ, Σ)] :
                                    C[:, tl_pick_idx(μ, sqrt.(max.(diag(Σ), 0.0)), yb, t)]
                    if method == :tl_mean
                        dg[:dsd] = sqrt(exp(θ[1]))          # 残差幅度 / 源增益标准差
                    else
                        dg[:rho] = 1/(1 + exp(-θ[4])); dg[:a] = exp(θ[5]); dg[:snS] = exp(θ[6])
                    end
                end
                ev!(xn)
            end
        else
            error("未知迁移方法 $(method)")
        end
    catch e
        e isa TLBudgetHit || rethrow()
    end
    length(curve) < B && append!(curve, fill(best[], B - length(curve)))   # NM 提前收敛: 补平
    if polish > 0
        cap[] = nev[] + polish
        try
            optimize(z -> ev!(clamp.(z, 0.0, 1.0) .* xs), bu[] ./ xs,
                     NelderMead(initial_simplex=Optim.AffineSimplexer(0.02, 0.0)),
                     Optim.Options(iterations=100000, g_tol=1e-12))
        catch e
            e isa TLBudgetHit || rethrow()
        end
    end
    y_zs = (method in (:tl_nm, :tl_init, :tl_mean, :tl_mtgp) && length(Y) >= 2) ? Y[2] : NaN
    return (ubest=bu[], ybest=best[], y0=y0, curve=curve, U=reduce(hcat, Ucols), Y=copy(Y),
            nev=nev[], diag=dg, y_zs=y_zs, t=time()-t0, t_fit=tfit)
end

"""组装第 p 层的源数据: 重标度 γ、换成增益单位、限量、算迁移设计点。"""
function tl_source_pool(srcs, p::Int, κt::Float64, seed::Int)
    rng = Random.MersenneTwister(seed)
    lo = [0.0, 0.0]; hi = [BOX_HI[1], BOX_HI[2]]; xs = hi .- lo
    keep = Vector{Vector{Float64}}(); keepy = Float64[]
    rest = Vector{Vector{Float64}}(); resty = Float64[]
    opts = Vector{Vector{Float64}}()
    for s in srcs
        Ls = s.layers[p]; ratio = s.κ / κt
        n = size(Ls.U, 2); gs = Ls.Y .- Ls.y0
        cols = [[Ls.U[1, i] * ratio, Ls.U[2, i]] for i in 1:n]
        chosen = Int[]
        for i in sortperm(gs)                                  # 好点优先, 但要互相分开
            all(j -> maximum(abs.((cols[i] .- cols[j]) ./ xs)) > 0.02, chosen) && push!(chosen, i)
            length(chosen) >= TL_SRC_TOPK && break
        end
        for i in 1:n
            if i in chosen
                push!(keep, cols[i]); push!(keepy, gs[i])
            else
                push!(rest, cols[i]); push!(resty, gs[i])
            end
        end
        push!(opts, clamp.([Ls.ubest[1] * ratio, Ls.ubest[2]], lo, hi))
    end
    nfill = max(TL_SRC_CAP - length(keep), 0)
    pick = Random.shuffle(rng, collect(1:length(rest)))[1:min(nfill, length(rest))]
    Ucols = vcat(keep, rest[pick]); yg = vcat(keepy, resty[pick])
    # 迁移设计点: 坐标中位数(Shaydulin) + KDE 密度最高、彼此分开的源最优
    O = reduce(hcat, opts) ./ xs
    med = [median(O[1, :]), median(O[2, :])]
    dens = [sum(exp.(-0.5 .* vec(sum(((O .- O[:, i]) ./ 0.06).^2; dims=1)))) for i in 1:size(O, 2)]
    pts = [med]
    for i in sortperm(dens, rev=true)
        length(pts) >= TL_N_TRANSFER && break
        all(c -> maximum(abs.(O[:, i] .- c)) > 0.06, pts) && push!(pts, O[:, i])
    end
    return (U=reduce(hcat, Ucols), yg=yg, pts=[clamp.(c .* xs, lo, hi) for c in pts], opts=opts)
end

# ── [tl_port] 覆盖式开局组合(portfolio / mode-covering transfer) ──
# 动机(实测): p=1 曲面是多峰的。GNN 预热下大多数实例 γ*≈2.2, 但 i6/i12/i20 的 γ*≈7–8、i11 的
#   γ*≈0.86; 冷启动下 i3/i20 的 β*≈0.42, 其余 0.9–2.8 —— 这几个恰好就是 T4 的离群实例。
#   中位数 + KDE 的 5 个开局点全被主峰内部 β 的散布占掉, 少数峰一个名额都分不到。
# 做法: 把目标实例看成"从源实例分布里抽出来的一个", 选 5 个开局点 P 使
#         E_{s∼w} [ min_{c∈P∪{原点}} regret_s(c) ]     最小,
#   regret_s(c) = g_s(c) − min g_s ≥ 0 是源实例 s 在 c 处相对自身最优的损失(R_a 单位),
#   g_s 用源实例自己第 p 层的数据拟合的 Matérn GP 均值(γ 先按 κ_s/κ_t 换到目标坐标)。
#   这是设施选址(facility location)问题, 目标函数是单调次模函数, 贪心有 (1−1/e) 近似保证。
#   和中位数的区别: 第 1 个点是"平均损失最小的折中点"(中位数的有原则版本); 之后每个点都去
#   覆盖"还没被覆盖、权重×损失最大"的那部分源实例 —— 少数峰只要有源实例在那里就会分到名额。
#   w_s(源权重): 等权(tl_port), 或按目标与源的特征相似度(tl_portp / tl_portg)。
"""返回 (pts, er, er_med, cov, cov_med, n_cand)。er = 组合的加权期望遗憾(预测), er_med = 同一目标函数下
   中位数+KDE 开局点(cmp_pts)的期望遗憾; cov = 遗憾 < 1e-3 的源权重占比。"""
function tl_portfolio(srcs, p::Int, κt::Float64, wv::Vector{Float64}, cmp_pts; npts::Int=TL_N_TRANSFER)
    lo = [0.0, 0.0]; hi = [BOX_HI[1], BOX_HI[2]]; xs = hi .- lo
    K = length(srcs)
    cand = Vector{Vector{Float64}}()
    for s in srcs                                       # 候选: 所有源实例在这一层评估过的点(换到目标坐标)
        Ls = s.layers[p]; ratio = s.κ / κt
        for i in 1:size(Ls.U, 2)
            push!(cand, clamp.([Ls.U[1, i] * ratio, Ls.U[2, i]], lo, hi))
        end
    end
    for h in halton_samples(200, lo, hi; skip=11)       # + 均匀铺一层, 防止源数据没覆盖到的区域被漏掉
        push!(cand, clamp.(h, lo, hi))
    end
    n_main = length(cand)
    append!(cand, [clamp.(c, lo, hi) for c in cmp_pts]) # 中位数 + KDE 点也作候选(放最后, 便于算对照)
    C = reduce(hcat, cand); nc = size(C, 2)
    G = zeros(K, nc); gb = zeros(K)
    for (k, s) in enumerate(srcs)
        Ls = s.layers[p]; ratio = s.κ / κt
        yg = Ls.Y .- Ls.y0                              # 增益(原点 = 0; 越负越好)
        gp = abl_build(:matern52, Ls.U, yg, s.θ[p], 2, xs)
        Cs = vcat(clamp.(C[1:1, :] ./ ratio, lo[1], hi[1]), C[2:2, :])   # 目标坐标 → 源坐标
        μ, _ = abl_pred(gp, Cs)
        G[k, :] .= μ
        gb[k] = min(minimum(μ), minimum(yg), 0.0)
    end
    R = max.(G .- gb, 0.0)                              # regret_s(c)
    w = wv ./ sum(wv)
    cur0 = -gb                                          # 只有原点时的遗憾(原点增益 = 0)
    cur = copy(cur0); chosen = Int[]
    for _ in 1:npts
        bv = 0.0; bj = 0
        for j in 1:nc
            j in chosen && continue
            v = 0.0
            @inbounds for k in 1:K
                v += w[k] * (cur[k] - min(cur[k], R[k, j]))
            end
            v > bv + 1e-12 && (bv = v; bj = j)
        end
        bj == 0 && break                                # 所有源都已覆盖: 剩下的名额交给 Halton
        push!(chosen, bj); cur .= min.(cur, R[:, bj])
    end
    cm = copy(cur0)
    for j in (n_main+1):nc
        cm .= min.(cm, R[:, j])
    end
    return (pts=[C[:, j] for j in chosen], er=dot(w, cur), er_med=dot(w, cm),
            cov=sum(w[cur .< 1e-3]), cov_med=sum(w[cm .< 1e-3]), n_cand=nc)
end

"""[tl_port 预计算] 一个 (源实例, 条件) 每层源 GP 的 Matérn 超参数(只依赖源数据, 与目标无关 ⇒ 算一次缓存)。
   GP 在源实例自己的坐标里拟合; 换到目标坐标只需在预测时把候选点的 γ 除以 κ_s/κ_t。"""
function tl_srcgp_task(tk)
    Random.seed!(tk.seed)
    xs = [BOX_HI[1], BOX_HI[2]]
    th = Vector{Vector{Float64}}()
    for L in tk.layers
        push!(th, abl_fit(:matern52, L.U, L.Y .- L.y0, 2, xs))
    end
    return th
end

"""源数据库: 一个 (实例, 条件) 的逐层链, Matérn BO + 二维 NM 精修。"""
function tl_source_task(tk)
    g = tk.g; f_full = tl_make_evf(g, tk.cond)
    ghi1 = hasproperty(tk, :ghi1) ? tk.ghi1 : BOX_HI[1]      # 退火分支源链: 第 1 层 γ ≤ ghi1
    ghia = hasproperty(tk, :ghiall) ? tk.ghiall : BOX_HI[1]   # 全层退火源链: 每层 γ ≤ ghia
    prefix = Float64[]; yprev = nothing; layers = NamedTuple[]; t0 = time()
    for p in 1:tk.p_max
        pre = copy(prefix)
        r = tl_layer(u -> f_full(vcat(pre, u)), :matern52, tk.budget, tk.seed + 101p,
                     tl_depth_dep(tk.cond) ? nothing : yprev, nothing;
                     polish=TL_SRC_POLISH, ghi=(p == 1 ? min(ghi1, ghia) : ghia))
        push!(layers, (p=p, U=r.U, Y=r.Y, y0=r.y0, ubest=r.ubest, ybest=r.ybest, nev=r.nev))
        prefix = vcat(prefix, r.ubest); yprev = r.ybest
    end
    return (inst=g.inst, cond=tk.cond, layers=layers, kmean=tl_kappa(g, "mean"),
            kmax=tl_kappa(g, "max"), t_total=time()-t0)
end

# ── 计数评估器(带缓存): 同一点重复评估不计次数, 与生产 hybrid 的 ev 缓存一致 ──
function tl_counter(f, hi::Vector{Float64}, cap::Base.RefValue{Int})
    cache = Dict{Vector{Float64},Float64}(); curve = Float64[]; nev = Ref(0)
    best = Ref(Inf); bx = Ref(Float64[])
    evx! = function (xx)
        xc = clamp.(xx, 0.0, hi); key = round.(xc, digits=9)
        haskey(cache, key) && return cache[key]
        nev[] >= cap[] && throw(TLBudgetHit())
        v = f(xc); nev[] += 1; cache[key] = v
        v < best[] && (best[] = v; bx[] = copy(xc))
        push!(curve, best[])
        return v
    end
    return evx!, curve, nev, best, bx, cache
end

"""从当前最优反复重启 NM(归一化坐标, 初始单纯形边长依次取 steps), 直到预算用完。
   一次重启没有产生任何新评估(例如所有坐标都贴在边界上)就停止, 防止死循环。"""
function tl_nm_cycle!(evx!, bx::Base.RefValue{Vector{Float64}}, xs::Vector{Float64}, steps,
                      nev::Base.RefValue{Int})
    try
        for st in Iterators.cycle(steps)
            n0 = nev[]
            optimize(z -> evx!(clamp.(z, 0.0, 1.0) .* xs), bx[] ./ xs,
                     NelderMead(initial_simplex=Optim.AffineSimplexer(st, 0.0)),
                     Optim.Options(iterations=100000, g_tol=1e-12))
            nev[] == n0 && break
        end
    catch e
        e isa TLBudgetHit || rethrow()
    end
    return nothing
end

"""[tl_hyb] hybrid 式前缀精修: 逐层轮扫(第 k 层二维 NM, 其余层固定, 每层 S 次)
   + 全维 NM 精修(Q 次)。只接受改进 ⇒ 单调。"""
function tl_refine_full(f_full, x0::Vector{Float64}, y0::Float64, p::Int, nsweep::Int, S::Int, Q::Int;
                        g1hi::Float64=BOX_HI[1], gallhi::Float64=BOX_HI[1])
    hi = repeat([gallhi, BOX_HI[2]], p); hi[1] = min(g1hi, gallhi); xs = copy(hi)   # g1hi: 第 1 层 γ 上界(tl_ann); gallhi: 每层 γ 上界(tl_annx)
    cap = Ref(0)
    evx!, curve, nev, best, bx, cache = tl_counter(f_full, hi, cap)
    best[] = y0; bx[] = copy(x0); cache[round.(clamp.(x0, 0.0, hi), digits=9)] = y0
    if p >= 2 && S > 0
        for s in 1:nsweep, k in 1:(p-1)
            cap[] = nev[] + S
            base = copy(bx[]); idx = (2k-1):(2k)
            try
                optimize(z -> begin
                             xx = copy(base); xx[idx] .= clamp.(z, 0.0, 1.0) .* xs[idx]
                             evx!(xx)
                         end,
                         base[idx] ./ xs[idx], NelderMead(initial_simplex=Optim.AffineSimplexer(0.05, 0.0)),
                         Optim.Options(iterations=100000, g_tol=1e-12))
            catch e
                e isa TLBudgetHit || rethrow()
            end
        end
    end
    y_sw = best[]
    if Q > 0
        cap[] = nev[] + Q
        tl_nm_cycle!(evx!, bx, xs, (0.03, 0.01), nev)
    end
    return (x=bx[], ybest=best[], curve=curve, nev=nev[], gain_sweep=y0 - y_sw, gain_polish=y_sw - best[])
end

"""[tl_full] Shaydulin 式整向量迁移: 每个深度 p 独立 —— 取其它实例 hybrid 最优向量
   (γ 按 κ 重标度)的坐标中位数 + KDE 高密度源向量作候选, 评估后从最好者做全维 NM
   (重启式)直到预算 tk.full_budget。不依赖逐层链, 也不需要源数据库。"""
function tl_full_task(tk, f_full, κt::Float64, t0::Float64)
    layers = NamedTuple[]
    for p in 1:tk.p_max
        hi = repeat([BOX_HI[1], BOX_HI[2]], p); xs = copy(hi)
        V = Vector{Vector{Float64}}()
        for s in tk.srcx
            haskey(s.x, p) || continue
            v = copy(s.x[p]); v[1:2:end] .*= s.κ / κt
            push!(V, clamp.(v, 0.0, hi))
        end
        isempty(V) && error("tl_full: 没有任何源实例的 p=$(p) 整向量(hybrid 基准缺这一层?)")
        Vn = reduce(hcat, V) ./ xs
        med = vec(median(Vn; dims=2))
        h = 0.08 * sqrt(p)
        dens = [sum(exp.(-0.5 .* vec(sum(abs2, Vn .- Vn[:, i]; dims=1)) ./ h^2)) for i in 1:size(Vn, 2)]
        cands = [med]
        for i in sortperm(dens, rev=true)
            length(cands) >= TL_N_TRANSFER && break
            all(c -> maximum(abs.(Vn[:, i] .- c)) > 0.06, cands) && push!(cands, Vn[:, i])
        end
        cap = Ref(tk.full_budget)
        evx!, curve, nev, best, bx, _ = tl_counter(f_full, hi, cap)
        y_zs = NaN
        try
            for (ci, c) in enumerate(cands)
                v = evx!(c .* xs)
                ci == 1 && (y_zs = v)
            end
        catch e
            e isa TLBudgetHit || rethrow()
        end
        tl_nm_cycle!(evx!, bx, xs, (0.05, 0.02, 0.01), nev)
        cv = copy(curve)
        length(cv) < tk.full_budget && append!(cv, fill(best[], tk.full_budget - length(cv)))
        push!(layers, (p=p, curve=cv, ybest=best[], ubest=bx[][end-1:end], xfull=copy(bx[]), nev=nev[],
                       diag=Dict{Symbol,Float64}(:n_src => length(V), :n_cand => length(cands)),
                       y_zs=y_zs, y0=NaN))
    end
    return (inst=tk.g.inst, cond=tk.cond, method=tk.method, rep=tk.rep, layers=layers,
            src_insts=tk.src_insts, t_total=time()-t0, t_fit=0.0)
end

"""[多起点] 新层 BO 结束后, 从 BO 评估过的点里挑 K+1 个**互相分开**的好点(最好点 + K 个别的盆地),
   各做一次只动新层的二维 NM(每个 B_ms 次), 取最好者交给后面的轮扫/全维精修。
   目的: p=1 的 β 是多峰的, 单起点精修会被困在错的峰里, 而这个误差会传给后面所有层。
   BO 已评估过的点放进缓存, 不重复计数。"""
function tl_multistart_newlayer(f_full, pre::Vector{Float64}, r, K::Int, B_ms::Int; ghi::Float64=BOX_HI[1])
    lo = [0.0, 0.0]; hi = [ghi, BOX_HI[2]]; xs = copy(hi)
    cap = Ref(0)
    evx!, curve, nev, best, bx, cache = tl_counter(u -> f_full(vcat(pre, u)), hi, cap)
    for i in 1:size(r.U, 2)
        cache[round.(clamp.(r.U[:, i], lo, hi), digits=9)] = r.Y[i]
    end
    best[] = r.ybest; bx[] = copy(r.ubest)
    starts = Vector{Vector{Float64}}()
    for i in sortperm(r.Y)
        u = clamp.(r.U[:, i], lo, hi)
        all(s -> maximum(abs.((u .- s) ./ xs)) > 0.08, starts) && push!(starts, u)
        length(starts) >= K + 1 && break
    end
    for s in starts
        cap[] = nev[] + B_ms
        try
            optimize(z -> evx!(clamp.(z, 0.0, 1.0) .* xs), s ./ xs,
                     NelderMead(initial_simplex=Optim.AffineSimplexer(0.03, 0.0)),
                     Optim.Options(iterations=100000, g_tol=1e-12))
        catch e
            e isa TLBudgetHit || rethrow()
        end
    end
    switched = maximum(abs.((bx[] .- r.ubest) ./ xs)) > 0.08          # 最终是否换到了别的盆地
    return (u=bx[], y=best[], curve=curve, nev=nev[], n_starts=length(starts), gain=r.ybest - best[],
            switched=switched)
end

"""[INTERP] Zhou et al., PRX 10, 021067 (2020) 的线性插值外推: 由深度 q 的 (γ1,β1,…,γq,βq) 得到深度 q+1 的整条向量
   x_i^(q+1) = (i−1)/q · x_{i−1}^(q) + (q−i+1)/q · x_i^(q),  i = 1..q+1,  x_0 = x_{q+1} = 0;  γ、β 各自插值。
   相当于把整条退火曲线在时间上"拉长一格", 所有层一起移动 —— 逐层追加做不到的整体调整。"""
function tl_interp(x::Vector{Float64}, q::Int)
    @assert length(x) == 2q
    ip(v) = [((i - 1) / q) * (i >= 2 ? v[i-1] : 0.0) + ((q - i + 1) / q) * (i <= q ? v[i] : 0.0) for i in 1:q+1]
    out = zeros(2(q + 1))
    out[1:2:end] .= ip(x[1:2:end]); out[2:2:end] .= ip(x[2:2:end])
    return out
end

"""目标运行: 一个 (实例, 条件, 方法, 重复)。逐层链(每层新层预算 tk.budget);
   tl_hyb 类在每层新层之后再做多起点 + 轮扫 + 全维精修; tl_full 不走链(见 tl_full_task)。
   tl_port*: 开局点换成覆盖式组合(tk.srcs 每项带 θ = 源 GP 超参数, w = 源权重);
   tl_hybx: 与 tl_hyb 完全相同, 只是 tk.srcs 来自另一种条件的源数据库(在主程序里组装)。"""
function tl_target_task(tk)
    g = tk.g; f_full = tl_make_evf(g, tk.cond); κt = tl_kappa(g, TL_RESCALE); t0 = time()
    tk.method == :tl_full && return tl_full_task(tk, f_full, κt, t0)
    prefix = Float64[]; yprev = nothing; layers = NamedTuple[]; tfit = 0.0
    for p in 1:tk.p_max
        pre = copy(prefix)
        # tl_hyb / tl_port* / tl_hybx 的新层 = tl_init(只差开局点或源数据库);
        # m52_hyb 的新层 = matern52(不用任何迁移), 其余流程与 tl_hyb 完全相同
        hyblike = tk.method in TL_HYB_LIKE
        lm = tk.method == :m52_hyb ? :matern52 : (hyblike ? :tl_init : tk.method)
        srcp = lm == :matern52 ? nothing : tl_source_pool(tk.srcs, p, κt, tk.seed + 31p)
        pdg = Dict{Symbol,Float64}()
        if tk.method in TL_PORT_METHODS
            wv = Float64[s.w for s in tk.srcs]
            tp = time()
            pf = tl_portfolio(tk.srcs, p, κt, wv, srcp.pts)
            tfit += time() - tp
            srcp = merge(srcp, (pts = pf.pts,))          # 只换开局点; 其余(GP 先验数据等)不变
            pdg[:port_er] = pf.er; pdg[:port_er_med] = pf.er_med
            pdg[:port_cov] = pf.cov; pdg[:port_cov_med] = pf.cov_med
            pdg[:ess] = sum(wv)^2 / sum(abs2, wv)
        end
        srcp === nothing || (pdg[:n_pts] = length(srcp.pts))
        g1hi = tk.method in TL_ANN_M ? tk.ann_gmax : BOX_HI[1]        # tl_ann/tl_annx: 第 1 层 γ 始终 ≤ ann_gmax
        gall = tk.method == :tl_annx ? tk.ann_gmax : BOX_HI[1]        # tl_annx: 每一层 γ 都 ≤ ann_gmax
        ghi  = p == 1 ? min(g1hi, gall) : gall
        r = tl_layer(u -> f_full(vcat(pre, u)), lm, tk.budget, tk.seed + 101p,
                     tl_depth_dep(tk.cond) ? nothing : yprev, srcp; ghi=ghi)
        tfit += r.t_fit
        if hyblike
            u1 = r.ubest; y1 = r.ybest; ms_curve = Float64[]; ms_nev = 0
            dg = merge(copy(r.diag), pdg)
            if p <= tk.ms_p && tk.ms_k > 0                                   # 新层多起点(默认只在 p=1)
                ms = tl_multistart_newlayer(f_full, pre, r, tk.ms_k, tk.ms_b; ghi=ghi)
                u1 = ms.u; y1 = ms.y; ms_curve = ms.curve; ms_nev = ms.nev
                dg[:gain_ms] = ms.gain; dg[:ms_switched] = ms.switched ? 1.0 : 0.0
            end
            # INTERP 候选(QAOA_TL_INTERP 里列出的方法, p ≥ 2): 把上一深度精修后的整条向量按 INTERP 外推成 2p 维,
            # 只花 1 次演化; 比"前缀 + BO 新层"好就换它作为轮扫/精修的起点(所有层一起动, 不再锁住前缀)
            int_curve = Float64[]; int_nev = 0
            if p >= 2 && hasproperty(tk, :interp) && tk.interp
                hi_i = repeat([gall, BOX_HI[2]], p); hi_i[1] = min(g1hi, gall)
                xi = clamp.(tl_interp(pre, p - 1), 0.0, hi_i)
                yi = f_full(xi); int_nev = 1
                xb = vcat(pre, u1)
                dg[:interp_gain] = y1 - yi                        # >0: INTERP 比 BO 新层好
                dg[:interp_win]  = yi < y1 ? 1.0 : 0.0
                if yi < y1
                    xb = xi; y1 = yi
                end
                push!(int_curve, y1)
            else
                xb = vcat(pre, u1)
            end
            # 深层预算随 p 增长(QAOA_TL_SWEEP_PP / QAOA_TL_POLISH_PP > 0 时): 每层轮扫 max(S, S_pp·p), 全维精修 max(Q, Q_pp·p)
            Sp = hasproperty(tk, :sweep_pp)  ? max(tk.sweep,  tk.sweep_pp * p)  : tk.sweep
            Qp = hasproperty(tk, :polish_pp) ? max(tk.polish, tk.polish_pp * p) : tk.polish
            hr = tl_refine_full(f_full, xb, y1, p, tk.nsweep, Sp, Qp; g1hi=g1hi, gallhi=gall)
            dg[:gain_sweep] = hr.gain_sweep; dg[:gain_polish] = hr.gain_polish
            dg[:budget_S] = Sp; dg[:budget_Q] = Qp
            push!(layers, (p=p, curve=vcat(r.curve, ms_curve, int_curve, hr.curve), ybest=hr.ybest, ubest=hr.x[end-1:end],
                           xfull=copy(hr.x), nev=r.nev + ms_nev + int_nev + hr.nev, diag=dg, y_zs=r.y_zs, y0=r.y0))
            prefix = copy(hr.x); yprev = hr.ybest                   # 前缀被轮扫/精修改过
        else
            push!(layers, (p=p, curve=r.curve, ybest=r.ybest, ubest=r.ubest, xfull=vcat(pre, r.ubest),
                           nev=r.nev, diag=merge(copy(r.diag), pdg), y_zs=r.y_zs, y0=r.y0))
            prefix = vcat(prefix, r.ubest); yprev = r.ybest
        end
    end
    return (inst=g.inst, cond=tk.cond, method=tk.method, rep=tk.rep, layers=layers,
            src_insts=tk.src_insts, t_total=time()-t0, t_fit=tfit)
end
end # @everywhere begin (transfer_bo)

if length(ARGS) >= 1 && ARGS[1] == "transfer_bo"
    using Statistics, LinearAlgebra
    n_at        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    n_instances = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
    p_max       = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
    B_tgt       = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 30
    n_rep       = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2
    methods     = length(ARGS) >= 7 ? Symbol.(split(ARGS[7], ",")) : collect(TL_METHODS)
    K_src       = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : n_instances - 1
    B_src       = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 80
    # tl_full / tl_hyb 的预算(环境变量; 写进 checkpoint 文件名, 改了会重算)
    TL_FULL_B   = parse(Int, get(ENV, "QAOA_TL_FULL_BUDGET", "300"))   # tl_full: 每个深度的总演化数
    TL_SWEEP_B  = parse(Int, get(ENV, "QAOA_TL_SWEEP", "15"))          # tl_hyb: 每层每次轮扫的演化数
    TL_NSWEEP   = parse(Int, get(ENV, "QAOA_TL_NSWEEP", "1"))          # tl_hyb: 轮扫遍数
    TL_POLISH_B = parse(Int, get(ENV, "QAOA_TL_POLISH", "60"))         # tl_hyb: 每层全维精修演化数
    TL_SWEEP_PP  = parse(Int, get(ENV, "QAOA_TL_SWEEP_PP", "0"))       # >0: 深度 p 时每层轮扫 = max(QAOA_TL_SWEEP, 这个值×p)
    TL_POLISH_PP = parse(Int, get(ENV, "QAOA_TL_POLISH_PP", "0"))      # >0: 深度 p 时全维精修 = max(QAOA_TL_POLISH, 这个值×p)
    (TL_SWEEP_PP >= 0 && TL_POLISH_PP >= 0) || error("QAOA_TL_SWEEP_PP / QAOA_TL_POLISH_PP 必须 ≥ 0")
    TL_INTERP_M = Symbol[Symbol(strip(x)) for x in split(get(ENV, "QAOA_TL_INTERP", ""), ",") if !isempty(strip(x))]   # 加 INTERP 候选的方法
    TL_MS_K     = parse(Int, get(ENV, "QAOA_TL_MS_K", "2"))            # tl_hyb/m52_hyb: 新层多起点的额外盆地数(0=关闭)
    TL_MS_B     = parse(Int, get(ENV, "QAOA_TL_MS_B", "30"))           # 每个起点的二维 NM 演化数
    TL_MS_P     = parse(Int, get(ENV, "QAOA_TL_MS_P", "1"))            # 只在 p ≤ 这个值的层做多起点
    n_grid      = 60
    optimizer   = "zygote"
    TL_CONDS    = String.(split(get(ENV, "QAOA_TL_CONDS", "cold_start,gnn_warmstart"), ","))   # 可只跑一种条件
    all(c -> c in ("cold_start", "gnn_warmstart", "gnn_bias"), TL_CONDS) ||
        error("QAOA_TL_CONDS 只能是 cold_start / gnn_warmstart / gnn_bias")
    ("gnn_bias" in TL_CONDS) && TL_BIAS_MODE == :egeo &&
        println("gnn_bias(B3): 初态 = 完整 GNN 预热态; 混合层 = (1−w_k)·Egger 精确 mixer + w_k·标准 mixer, w_k = 1 − $(TL_BIAS_R)^(k−1)(只依赖层号)")
    ("gnn_bias" in TL_CONDS) && TL_BIAS_MODE != :egeo && println(TL_BIAS_MODE == :interp ?
        "gnn_bias(B2): 初态 = 完整 GNN 预热态; 混合层从 Egger 精确版定制 mixer 线性过渡到标准 mixer (w=(k−1)/(p−1))" :
        "gnn_bias(B1-$(TL_BIAS_MODE)): 全幅横场 + GNN 偏置, λ_k = " *
        (TL_BIAS_MODE == :lin ? "$(TL_BIAS_L0)·(1−(k−1)/p)" : "$(TL_BIAS_L0)·$(TL_BIAS_R)^(k−1)") *
        ", 初态 = " * (TL_BIAS_PREP ? "第 1 层混合层的本征态" : "冷启动 |+⟩"))
    for m in methods
        m in TL_METHODS || error("未知方法 $(m); 可选: $(join(TL_METHODS, ","))")
    end
    TL_RESCALE in ("mean", "max", "none") || error("QAOA_TL_RESCALE 只能是 mean / max / none")
    TL_MC_CH    = Symbol.(split(get(ENV, "QAOA_TL_MC", "tl_hyb,tl_port,tl_portp"), ","))   # tl_mc 的组成链
    TL_MC_M     = length(TL_MC_CH)                                     # tl_hyb3 的预算倍数 = 链数
    TL_VIRT = Dict(:tl_mc => TL_MC_CH, :tl_mcs => [:tl_hyb, :tl_hybs2, :tl_hybs3],
                   :tl_mcm => [:tl_hyb, :tl_port, :m52_hyb], :tl_mca => vcat(TL_MC_CH, [:tl_ann]), :tl_mcx => vcat(TL_MC_CH, [:tl_ann, :tl_annx]))
    if :tl_mc in methods
        all(m -> m in TL_HYB_LIKE && m != :tl_hyb3 && m != :tl_hybx, TL_MC_CH) ||
            error("QAOA_TL_MC 的组成链只能是 tl_hyb / tl_port / tl_portp / tl_portg / m52_hyb / tl_hybs2 / tl_hybs3")
    end
    for vm in TL_VIRT_M
        vm in methods || continue
        all(m -> m in methods, TL_VIRT[vm]) ||
            error("$(vm) 需要把它的组成链 $(join(TL_VIRT[vm], ",")) 也放进方法列表(已算过的会直接从 checkpoint 读入)")
    end
    for m in TL_INTERP_M
        m in TL_HYB_LIKE || error("QAOA_TL_INTERP 只能列 tl_hyb 类的链方法(tl_hyb, tl_port, tl_portp, tl_ann, ...), 不能是 $(m)")
    end
    TL_PORT_H   = parse(Float64, get(ENV, "QAOA_TL_PORT_H", "0.5"))      # tl_portp/tl_portg: 特征相似度核的带宽(标准化单位)
    TL_PORT_H > 0 || error("QAOA_TL_PORT_H 必须 > 0")
    if :tl_hybx in methods
        "gnn_warmstart" in TL_CONDS || error("tl_hybx 只对 gnn_warmstart 目标有定义(用冷启动的源数据库), QAOA_TL_CONDS 里要有 gnn_warmstart")
    end
    1 <= p_max <= 20 || error("p_max 只能取 1..20")
    p_max > 5 && println("p_max=$(p_max) > 5: hybrid 基准只到 p=5, p>5 时表格里的参照改为各方法运行中的最好值(见表 Deep)")
    TL_ANN_GMAX = parse(Float64, get(ENV, "QAOA_TL_ANN_GMAX", "1.0"))  # tl_ann: 第 1 层 γ 的上界(μs)
    B_tgt >= 8 || error("每层预算至少 8(1 个原点 + 5 个迁移点 + 至少 2 步 BO)")
    1 <= K_src <= n_instances - 1 || error("源个数 K 必须在 1..$(n_instances-1)")
    B_src >= 30 || error("源预算至少 30")

    println("\n" * "="^70)
    println("跨实例迁移 GP (留一法: 目标实例绝不出现在自己的源里)")
    println("n=$(n_at)  实例数=$(n_instances)  p=1..$(p_max)  每层预算 B=$(B_tgt) 次演化  重复=$(n_rep)")
    println("方法: $(join(methods, ", "))")
    println("源: 每个目标用 K=$(K_src) 个其它实例; 源数据库每层 Matérn BO $(B_src) 次 + NM 精修 $(TL_SRC_POLISH) 次")
    println("γ 重标度: $(TL_RESCALE)  (" * (TL_RESCALE == "mean" ? "Shaydulin: γ·mean J 不变" :
            TL_RESCALE == "max" ? "Montañez-Barrera: γ·max J 不变" : "不重标度, 消融对照") * ")")
    println("条件: $(join(TL_CONDS, ", "))   p=1 标准答案: $(n_grid)×$(n_grid) 网格 + NM 精修")
    TL_WS_MIXSIGN < 0 && println("GNN 预热使用 Egger 精确版定制 mixer(失谐 −Ωt·q_i, 初态是 mixer 的本征态); hybrid 基准是原 mixer 算的")
    TL_WS_EPS > 0 && println("预热正则化 ε = $(TL_WS_EPS): GNN 概率夹到 [ε, 1−ε]  (注意: hybrid 基准是 ε=0 算的 —— " *
                             "T3 里 gnn 的'超过 hybrid'就是 ε 突破了原来的天花板)")
    :tl_hyb in methods && println("tl_hyb : 每层 = 新层 tl_init $(B_tgt) 次 + 轮扫 $(TL_NSWEEP) 遍×(p−1) 层×$(TL_SWEEP_B) 次 + 全维精修 $(TL_POLISH_B) 次" *
                                  (TL_MS_K > 0 ? "; p≤$(TL_MS_P) 的新层另做 $(TL_MS_K+1) 起点×$(TL_MS_B) 次的多盆地精修" : ""))
    any(m -> m in TL_PORT_METHODS, methods) &&
        println("tl_port: 开局 $(TL_N_TRANSFER) 点 = 覆盖式组合(最小化源实例分布上的期望遗憾, 贪心设施选址), 其余同 tl_hyb;" *
                " 源权重: tl_port 等权 / tl_portp 物理特征(w_cv, unwanted, fidelity) / tl_portg GNN 特征" *
                "(conf_mean, entropy, ws_ecut, gnn_round), 相似度核带宽 h=$(TL_PORT_H)")
    :tl_mc in methods && println("tl_mc  : 多链迁移 = $(join(TL_MC_CH, " + ")) 各一条链(开局不同), 每个深度取最好; 演化数为各链之和")
    :tl_ann in methods && println("tl_ann : 退火分支链: 第 1 层 γ ≤ $(TL_ANN_GMAX) μs (全程, 含轮扫/精修), 源 = 其它实例的退火分支源链(同条件); 其余同 tl_hyb")
    :tl_mca in methods && println("tl_mca : tl_mc 的 3 条链 + tl_ann, 每个深度取最好")
    :tl_annx in methods && println("tl_annx: 全层退火链: 每一层 γ ≤ $(TL_ANN_GMAX) μs (全程), 源 = 全层受限的退火源链(同条件); 其余同 tl_ann")
    :tl_mcx in methods && println("tl_mcx : tl_mca 的链 + tl_annx, 每个深度取最好")
    (TL_SWEEP_PP > 0 || TL_POLISH_PP > 0) &&
        println("深层预算随 p 增长: 每层轮扫 = max($(TL_SWEEP_B), $(TL_SWEEP_PP)·p), 全维精修 = max($(TL_POLISH_B), $(TL_POLISH_PP)·p)" *
                " (p=10 时轮扫每层 $(max(TL_SWEEP_B, 10TL_SWEEP_PP))、精修 $(max(TL_POLISH_B, 10TL_POLISH_PP)));" *
                " 两者都不超过原值的深度上链与原设置逐位一致 —— 表 PP 与原预算的 checkpoint 逐实例对比")
    isempty(TL_INTERP_M) || println("INTERP 候选: $(join(TL_INTERP_M, ", ")) 在 p≥2 时把上一深度的整条向量 INTERP 外推(1 次演化), " *
                                    "比'前缀+BO 新层'好就换作轮扫/精修起点 —— 表 PP 另与'同预算、不加 INTERP'配对, 表 I 统计换用次数")
    :tl_mcs in methods && println("tl_mcs : 对照 = tl_hyb 的 3 个随机种子(开局相同), 每个深度取最好 —— 多样性只来自种子")
    :tl_mcm in methods && println("tl_mcm : tl_hyb + tl_port + m52_hyb(第 3 条链不用迁移), 每个深度取最好")
    :tl_hyb3 in methods && println("tl_hyb3: tl_hyb 的所有预算 ×$(TL_MC_M)(新层 BO $(TL_MC_M*B_tgt)、多起点每点 $(TL_MC_M*TL_MS_B)、" *
                                   "轮扫每层 $(TL_MC_M*TL_SWEEP_B)、精修 $(TL_MC_M*TL_POLISH_B)) —— tl_mc 的等预算对照")
    :tl_hybx in methods && println("tl_hybx: 与 tl_hyb 相同, 但 gnn_warmstart 目标用**冷启动**的源数据库(检验迁移库是否必须与预热条件匹配)")
    :tl_full in methods && println("tl_full: 每个深度独立, 整向量迁移(hybrid 最优向量的中位数 + KDE 候选) + 全维 NM, 共 $(TL_FULL_B) 次")
    QAOA_BOX_SCALE != 1.0 && println("参数盒放大 ×$(QAOA_BOX_SCALE)")
    QAOA_OMEGA_T != 2.0 && println("混合层 Ωt = $(QAOA_OMEGA_T) (方法二默认 2)。若 QAOA_ABL_BASELINE 指向 Ωt=2 的 hybrid(目录名不带 _om):" *
                                   " 表 T3/T4 的'与 hybrid 的差距'表示'本 Ωt vs Ωt=2 的 hybrid', <0 即本 Ωt 更好, 且它不参与标准答案;" *
                                   " 指向同一 Ωt 的 hybrid(目录名带 _om$(QAOA_OMEGA_T))时则是正常的同曲面对比")
    println("="^70)

    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" *
                 (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    isfile(prior_file) || error("找不到先验 $(prior_tag) —— 先跑一次 gnn_batch_qaoa_weighted 生成它")
    gnn_params = deserialize(prior_file)
    GRAPHS, INST_FPS, INST_SETFP, INST_FROM_FILE =
        obtain_test_instances(n_at, n_instances, gnn_params; verbose=false, save=true, prior_tag=prior_tag)
    boxtag  = QAOA_BOX_SCALE == 1.0 ? "" : "_box$(QAOA_BOX_SCALE)"
    wstag   = (TL_WS_EPS > 0 ? "_ws$(TL_WS_EPS)" : "") * (TL_WS_MIXSIGN < 0 ? "_wsE" : "")   # ε>0 / Egger 精确 mixer: gnn 曲面变了 ⇒ 单独的目录
    omtag   = QAOA_OMEGA_T == 2.0 ? "" : "_om$(QAOA_OMEGA_T)"  # Ωt≠2: 两种条件的曲面都变了 ⇒ 单独的目录
    biastag = !("gnn_bias" in TL_CONDS) ? "" :
              TL_BIAS_MODE == :interp ? "_bint" :
              TL_BIAS_MODE == :egeo ? "_beg$(TL_BIAS_R)" :
              TL_BIAS_MODE == :lin ? "_blin$(TL_BIAS_L0)" * (TL_BIAS_PREP ? "" : "c") :
              "_bl$(TL_BIAS_L0)r$(TL_BIAS_R)" * (TL_BIAS_PREP ? "" : "c")          # B 方案参数 ⇒ 单独的目录
    src_dir = joinpath(directoryBasis, "ckpt_tlsrc_n$(n_at)_S$(B_src)" * boxtag * wstag * omtag * biastag)   # 层数够就复用(前几层确定性一致)
    tl_cfg  = "tl_n$(n_at)_S$(B_src)_P$(p_max)_T$(B_tgt)_K$(K_src)_$(TL_RESCALE)" * boxtag * wstag * omtag * biastag
    tl_dir  = joinpath(directoryBasis, "ckpt_" * tl_cfg)
    mkpath(src_dir); mkpath(tl_dir)
    check_ckpt_fingerprints(src_dir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE)
    check_ckpt_fingerprints(tl_dir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE)

    # ── 实例特征(只用实例本身, 不看任何优化结果) → tl_portp / tl_portg 的源权重 ──
    #   物理/嵌入: w_cv = 边权变异系数(权重 = 原子间距决定的耦合); unwanted = 嵌入引入的非目标耦合;
    #              fidelity = 嵌入保真度。离线分析: 落进少数峰的实例 w_cv 显著偏高(MWU p=0.002),
    #              unwanted/fidelity 也显著(p=0.016); 注意这是事后挑选的特征, 本实验就是它的检验。
    #   GNN 输出: conf_mean = 平均置信度 |2p−1|; entropy = 平均二元熵; ws_ecut = 预热乘积态的期望割/C_max;
    #              gnn_round = GNN 硬判决的割/C_max。离线分析: 这些与峰的归属无关(p>0.29) —— 作为消融。
    TL_FEAT_P = ("w_cv", "unwanted", "fidelity")
    TL_FEAT_G = ("conf_mean", "entropy", "ws_ecut", "gnn_round")
    function tl_feats(g)
        w = g.weights; pg = g.best_p_gnn
        conf = abs.(pg .- 0.5) .* 2
        pc = clamp.(pg, 1e-12, 1 - 1e-12)
        ent = -mean(pc .* log.(pc) .+ (1 .- pc) .* log.(1 .- pc))
        ecut = 0.0; rcut = 0.0
        for (e, wi) in zip(g.edges, w)
            a, b = e[1], e[2]
            ecut += wi * (pg[a]*(1 - pg[b]) + pg[b]*(1 - pg[a]))
            ((pg[a] > 0.5) != (pg[b] > 0.5)) && (rcut += wi)
        end
        phys = [std(w; corrected=false) / mean(w),
                hasproperty(g, :unwanted) ? Float64(g.unwanted) : NaN,
                hasproperty(g, :fidelity) ? Float64(g.fidelity) : NaN]
        return phys, [mean(conf), ent, ecut / g.cmax, rcut / g.cmax]
    end
    FEATP = Dict{Int,Vector{Float64}}(); FEATG = Dict{Int,Vector{Float64}}()
    for g in GRAPHS
        FEATP[g.inst], FEATG[g.inst] = tl_feats(g)
    end
    if :tl_portp in methods
        all(g -> all(isfinite, FEATP[g.inst]), GRAPHS) ||
            error("tl_portp 需要实例的 unwanted / fidelity 字段(从实例文件读入时才有)")
    end
    function tl_zscore(F::Dict{Int,Vector{Float64}})
        M = reduce(hcat, [F[g.inst] for g in GRAPHS])
        μ = vec(mean(M; dims=2)); σ = vec(std(M; dims=2, corrected=false))
        Dict(g.inst => [σ[i] > 1e-12 ? (F[g.inst][i] - μ[i]) / σ[i] : 0.0 for i in eachindex(μ)] for g in GRAPHS)
    end
    ZP = tl_zscore(FEATP); ZG = tl_zscore(FEATG)
    # 源权重 w_s = exp(−‖z_t − z_s‖² / (2 h² n_f)), 再设下限 0.05·max(任何源都不会被完全丢掉)
    function tl_wvec(t::Int, chosen::Vector{Int}, m::Symbol)
        m == :tl_port && return ones(length(chosen))
        Z = m == :tl_portp ? ZP : ZG
        nf = length(Z[t])
        w = [exp(-sum(abs2, Z[t] .- Z[s]) / (2 * TL_PORT_H^2 * nf)) for s in chosen]
        return max.(w, 0.05 * maximum(w))
    end

    # ── 阶段 0: p=1 网格标准答案(先找核消融目录里现成的) ──
    GRID = Dict{Tuple{Int,String},Float64}()
    for d in filter(x -> startswith(x, "ckpt_abl_n$(n_at)_") && occursin("_G$(n_grid)", x) &&
                         (QAOA_BOX_SCALE == 1.0 ? !occursin("_box", x) : occursin(boxtag, x)),
                    readdir(directoryBasis))
        dd = joinpath(directoryBasis, d)
        isfile(joinpath(dd, "instances_fingerprint.txt")) || continue
        ok = try
            check_ckpt_fingerprints(dd, GRAPHS, INST_FPS; from_file=INST_FROM_FILE, write=false); true
        catch
            false
        end
        ok || continue
        for g in GRAPHS, c in TL_CONDS
            ((TL_WS_EPS > 0 || TL_WS_MIXSIGN < 0) && c == "gnn_warmstart") && continue      # 核消融的网格是 ε=0、原 mixer 算的, 不能复用
            QAOA_OMEGA_T != 2.0 && continue                          # 核消融的网格是 Ωt=2 算的, 不能复用
            f = joinpath(dd, "t_i$(g.inst)_$(c)_p1_grid_r0.jls")
            (haskey(GRID, (g.inst, c)) || !isfile(f)) && continue
            r = try deserialize(f) catch; nothing end
            r === nothing || (GRID[(g.inst, c)] = -r.ybest / g.cmax)
        end
    end
    if QAOA_OMEGA_T != 2.0                              # Ωt≠2: omega_scan 已用同一算法(60×60 网格 + NM)算过 p=1 标准答案
        omd = joinpath(directoryBasis, "ckpt_omscan_n$(n_at)_G$(n_grid)" * boxtag)
        if isdir(omd) && (try check_ckpt_fingerprints(omd, GRAPHS, INST_FPS; from_file=INST_FROM_FILE, write=false); true catch; false end)
            for g in GRAPHS, c in TL_CONDS
                ((TL_WS_EPS > 0 || TL_WS_MIXSIGN < 0) && c == "gnn_warmstart") && continue  # omega_scan 的网格是 ε=0、原 mixer 算的
                f = joinpath(omd, "om_i$(g.inst)_$(c)_W$(QAOA_OMEGA_T).jls")
                (haskey(GRID, (g.inst, c)) || !isfile(f)) && continue
                r = try deserialize(f) catch; nothing end
                r === nothing || (GRID[(g.inst, c)] = r.ra)
            end
        end
    end
    for g in GRAPHS, c in TL_CONDS                     # 本模式自己算过的
        f = joinpath(src_dir, "grid_i$(g.inst)_$(c).jls")
        (haskey(GRID, (g.inst, c)) || !isfile(f)) && continue
        r = try deserialize(f) catch; nothing end
        r === nothing || (GRID[(g.inst, c)] = r)
    end
    println("p=1 网格标准答案: 现成 $(length(GRID)) 个, 需新算 $(2*length(GRAPHS) - length(GRID)) 个")

    # ── 阶段 0+1: 网格(缺的) + 源数据库, 一起派发 ──
    SRC = Dict{Tuple{Int,String},Any}()
    P1 = NamedTuple[]
    SRC_CONDS = unique(vcat(TL_CONDS, :tl_hybx in methods ? ["cold_start"] : String[]))   # tl_hybx 要冷启动的库
    for g in GRAPHS, c in SRC_CONDS
        (c in TL_CONDS && !haskey(GRID, (g.inst, c))) && push!(P1, (kind=:grid, g=g, cond=c, p_max=1, budget=0, seed=0))
        f = joinpath(src_dir, "src_i$(g.inst)_$(c).jls")
        r = isfile(f) ? (try deserialize(f) catch; nothing end) : nothing
        (r !== nothing && length(r.layers) < p_max) && (r = nothing)    # 层数不够: 重算
        if r === nothing
            push!(P1, (kind=:src, g=g, cond=c, p_max=p_max, budget=B_src,
                       seed=300000 + 1000*g.inst + tl_coff(c)))
        else
            SRC[(g.inst, c)] = r
        end
    end
    # tl_ann 的源: 每个冷启动实例再建一条"退火分支"源链(第 1 层 γ ≤ TL_ANN_GMAX), 保证分支匹配的迁移有足够的源。
    #   (普通源链在 Ωt=15.7 冷启动下第 1 层大多落在大 γ 盆地, 能用的小 γ 源往往只有 0–1 个)
    SRCA = Dict{Tuple{Int,String},Any}()                    # (实例, 条件) → 退火分支源链
    if :tl_ann in methods
        for g in GRAPHS, c in TL_CONDS
            f = joinpath(src_dir, "srcann_i$(g.inst)_$(c)_G$(TL_ANN_GMAX).jls")
            r = isfile(f) ? (try deserialize(f) catch; nothing end) : nothing
            (r !== nothing && length(r.layers) < p_max) && (r = nothing)
            if r === nothing
                push!(P1, (kind=:srcann, g=g, cond=c, p_max=p_max, budget=B_src,
                           seed=300000 + 1000*g.inst + tl_coff(c), ghi1=TL_ANN_GMAX))
            else
                SRCA[(g.inst, c)] = r
            end
        end
    end
    SRCAX = Dict{Tuple{Int,String},Any}()                   # (实例, 条件) → 全层退火源链(tl_annx 用)
    if :tl_annx in methods
        for g in GRAPHS, c in TL_CONDS
            f = joinpath(src_dir, "srcannx_i$(g.inst)_$(c)_G$(TL_ANN_GMAX).jls")
            r = isfile(f) ? (try deserialize(f) catch; nothing end) : nothing
            (r !== nothing && length(r.layers) < p_max) && (r = nothing)
            if r === nothing
                push!(P1, (kind=:srcannx, g=g, cond=c, p_max=p_max, budget=B_src,
                           seed=300000 + 1000*g.inst + tl_coff(c), ghi1=TL_ANN_GMAX, ghiall=TL_ANN_GMAX))
            else
                SRCAX[(g.inst, c)] = r
            end
        end
    end
    @printf("阶段 1(网格 + 源数据库): 共 %d 个任务待算, worker=%d\n", length(P1), nworkers())
    t_p1 = time()
    sdir = src_dir
    R1 = isempty(P1) ? Any[] : pmap(P1) do tk
        if tk.kind == :grid
            evf = tl_make_evf(tk.g, tk.cond)
            rg = abl_grid_truth(evf, [0.0, 0.0], [BOX_HI[1], BOX_HI[2]], 60)
            v = -rg.ybest
            try serialize(joinpath(sdir, "grid_i$(tk.g.inst)_$(tk.cond).jls"), v) catch; end
            wlog(@sprintf("    [迁移/网格] inst%d %-13s p=1 网格最优 R_a=%.4f", tk.g.inst, tk.cond, v), :always)
            (kind=:grid, inst=tk.g.inst, cond=tk.cond, val=v)
        else
            r = tl_source_task(tk)
            fn = tk.kind == :srcann  ? "srcann_i$(tk.g.inst)_$(tk.cond)_G$(tk.ghi1).jls" :
                 tk.kind == :srcannx ? "srcannx_i$(tk.g.inst)_$(tk.cond)_G$(tk.ghi1).jls" : "src_i$(tk.g.inst)_$(tk.cond).jls"
            try serialize(joinpath(sdir, fn), r) catch; end
            wlog(@sprintf("    [迁移/%s] inst%d %-13s R_a(p=1..%d) = %s  [%.1f 分钟]",
                          tk.kind == :srcann ? "退火源" : tk.kind == :srcannx ? "全层退火源" : "源",
                          tk.g.inst, tk.cond, tk.p_max, join((@sprintf("%.4f", -L.ybest) for L in r.layers), " "),
                          r.t_total/60), :always)
            (kind=tk.kind, inst=tk.g.inst, cond=tk.cond, val=r)
        end
    end
    for r in R1
        if r.kind == :grid
            GRID[(r.inst, r.cond)] = r.val
        elseif r.kind == :srcann
            SRCA[(r.inst, r.cond)] = r.val
        elseif r.kind == :srcannx
            SRCAX[(r.inst, r.cond)] = r.val
        else
            SRC[(r.inst, r.cond)] = r.val
        end
    end
    @printf("阶段 1 完成, 墙钟 %.1f 分钟\n", (time()-t_p1)/60)

    # ── 阶段 1b(只有 tl_port* 需要): 每个 (源实例, 条件, 层) 的源 GP 超参数, 算一次缓存 ──
    SRCGP = Dict{Tuple{Int,String},Vector{Vector{Float64}}}()
    if any(m -> m in TL_PORT_METHODS, methods)
        PG = NamedTuple[]
        for g in GRAPHS, c in TL_CONDS
            f = joinpath(src_dir, "srcgp_i$(g.inst)_$(c).jls")
            th = isfile(f) ? (try deserialize(f) catch; nothing end) : nothing
            (th !== nothing && length(th) < length(SRC[(g.inst, c)].layers)) && (th = nothing)
            if th === nothing
                push!(PG, (inst=g.inst, cond=c, layers=SRC[(g.inst, c)].layers,
                           seed=500000 + 1000*g.inst + tl_coff(c)))
            else
                SRCGP[(g.inst, c)] = th
            end
        end
        t_pg = time()
        RG = isempty(PG) ? Any[] : pmap(PG) do tk
            th = tl_srcgp_task(tk)
            try serialize(joinpath(sdir, "srcgp_i$(tk.inst)_$(tk.cond).jls"), th) catch; end
            (inst=tk.inst, cond=tk.cond, th=th)
        end
        for r in RG
            SRCGP[(r.inst, r.cond)] = r.th
        end
        @printf("阶段 1b(源 GP 超参数): 新算 %d 个, 复用 %d 个, 墙钟 %.1f 分钟\n",
                length(PG), length(SRCGP) - length(PG), (time()-t_pg)/60)
    end

    # ── 基准(hybrid): 最优值、最优整向量(tl_full 的一种源)、每个种子到深度 p 的累计演化数 ──
    base_name = get(ENV, "QAOA_ABL_BASELINE", "")
    # 基准是否与本次同一个 Ωt(目录名带 _om<Ωt> ⇔ 该 Ωt; 不带 ⇔ Ωt=2)。同 Ωt 才参与标准答案。
    BASE_SAME_OM = QAOA_OMEGA_T == 2.0 ? !occursin("_om", base_name) : occursin("_om$(QAOA_OMEGA_T)", base_name)
    # GNN 预热的曲面还取决于 mixer 符号和 ε: 基准目录名带 _wsE / _ws<ε> ⇔ 对应设置。不一致时 gnn 的基准不计入标准答案。
    BASE_SAME_WS = (occursin("_wsE", base_name) == (TL_WS_MIXSIGN < 0)) &&
                   (TL_WS_EPS > 0 ? occursin("_ws$(TL_WS_EPS)", base_name) : !occursin(r"_ws[0-9]", base_name))
    BASE  = Dict{Tuple{Int,String,Int},Float64}()          # (inst, cond, p) => 多种子中最好的 R_a
    BASEX = Dict{Tuple{Int,String,Int},Vector{Float64}}()  # 对应的最优角度向量(旧 checkpoint 没有 outx 字段)
    BASEN = Dict{Tuple{Int,String,Int},Vector{Int}}()      # 每个种子一条: 跑到深度 p 共用多少次演化
    if !isempty(base_name)
        bdir = joinpath(directoryBasis, base_name)
        if isdir(bdir)
            check_ckpt_fingerprints(bdir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE, write=false)
            bstat = Dict("文件" => 0, "outx" => 0, "trace" => 0, "logbuf" => 0)
            bfields = Ref{Any}(nothing)
            re_ev = r"inst(\d+)/p=(\d+)\s+(\S+)\s+ar=.*?\[(\d+)次演化\]"
            for g in GRAPHS, f in filter(x -> startswith(x, "task_i$(g.inst)_s") && endswith(x, ".jls"), readdir(bdir))
                r = try deserialize(joinpath(bdir, f)) catch; nothing end
                r === nothing && continue
                bstat["文件"] += 1
                bfields[] === nothing && (bfields[] = propertynames(r))
                has_x = hasproperty(r, :outx); has_t = hasproperty(r, :trace) && !isempty(r.trace)
                has_x && (bstat["outx"] += 1); has_t && (bstat["trace"] += 1)
                # 每个 (条件, p) 本层用了多少次演化: 优先 trace, 退而求其次从 logbuf 的汇总行解析
                per = Dict{Tuple{String,Int},Int}()
                if has_t
                    for e in r.trace
                        per[(e.condition, e.p)] = get(per, (e.condition, e.p), 0) + e.n_evals
                    end
                elseif hasproperty(r, :logbuf)
                    for ln in r.logbuf
                        m = match(re_ev, ln)
                        (m === nothing || parse(Int, m[1]) != g.inst) && continue
                        per[(String(m[3]), parse(Int, m[2]))] = parse(Int, m[4])
                    end
                    isempty(per) || (bstat["logbuf"] += 1)
                end
                for c in TL_CONDS, p in 1:p_max
                    (haskey(r.out, c) && haskey(r.out[c], p)) || continue     # gnn_bias: hybrid 没有这个条件
                    k = (g.inst, c, p)
                    if r.out[c][p] > get(BASE, k, -Inf)
                        BASE[k] = r.out[c][p]
                        (has_x && haskey(r.outx[c], p)) && (BASEX[k] = copy(r.outx[c][p]))
                    end
                    all(q -> haskey(per, (c, q)), 1:p) &&
                        push!(get!(BASEN, k, Int[]), sum(per[(c, q)] for q in 1:p))
                end
            end
            println("并入基准 $(base_name): $(length(BASE)) 个 (实例,条件,p) 值, 其中 $(length(BASEX)) 个带最优向量, " *
                    "$(length(BASEN)) 个带演化计数")
            println("  基准 checkpoint: $(bstat["文件"]) 个文件, 含 outx $(bstat["outx"]) 个, 含 trace $(bstat["trace"]) 个, " *
                    "演化数取自 logbuf $(bstat["logbuf"]) 个")
            bfields[] === nothing || println("  基准 checkpoint 字段: $(join(string.(bfields[]), ", "))")
        else
            println("⚠ QAOA_ABL_BASELINE=$(base_name) 目录不存在, 忽略")
        end
    end

    # ── tl_full 的源向量从哪来 ──
    #   hybrid : 其它实例在 hybrid 基准里的最优向量(Shaydulin 原意: 用"充分优化过"的参数)
    #   m52_hyb: 其它实例本次 m52_hyb 的结果 —— 每个实例独立优化、不用任何迁移, 所以对目标实例
    #            **没有任何信息泄漏**(推荐; 论文用这个)
    #   tl_hyb : 其它实例本次 tl_hyb 的精修结果。注意二阶泄漏: 源实例 s 的 tl_hyb 用过包括目标 t
    #            在内的源数据库做迁移种子, 所以 s 的结果间接带有 t 的信息(影响很弱, 但不干净)
    #   chain  : 其它实例源数据库逐层链的结果(质量最差: 逐层贪心, 深层差 hybrid 约 0.017)
    #   auto(默认): hybrid 有 outx 就用 hybrid, 否则 m52_hyb, 否则 tl_hyb, 再否则 chain
    TL_FULL_SRC = lowercase(get(ENV, "QAOA_TL_FULL_SRC", "auto"))
    full_src = TL_FULL_SRC
    if :tl_full in methods
        n_need = length(GRAPHS) * length(TL_CONDS) * p_max
        if full_src == "auto"
            full_src = (length(BASEX) == n_need && BASE_SAME_OM) ? "hybrid" :
                       (:m52_hyb in methods ? "m52_hyb" : (:tl_hyb in methods ? "tl_hyb" : "chain"))
        end
        full_src in ("hybrid", "m52_hyb", "tl_hyb", "chain") ||
            error("QAOA_TL_FULL_SRC 只能是 auto / hybrid / m52_hyb / tl_hyb / chain")
        full_src == "m52_hyb" && !(:m52_hyb in methods) && error("QAOA_TL_FULL_SRC=m52_hyb 需要把 m52_hyb 放进方法列表")
        (full_src == "hybrid" && !BASE_SAME_OM) &&
            error("hybrid 基准与本次的 Ωt 不同, 不能用它的角度向量作 tl_full 的源; 请设 QAOA_TL_FULL_SRC=m52_hyb 或 chain")
        full_src == "hybrid" && length(BASEX) < n_need &&
            error("QAOA_TL_FULL_SRC=hybrid, 但基准只有 $(length(BASEX))/$(n_need) 个最优向量(旧 checkpoint 没存 outx)")
        full_src == "tl_hyb" && !(:tl_hyb in methods) && error("QAOA_TL_FULL_SRC=tl_hyb 需要把 tl_hyb 放进方法列表")
        println("tl_full 的源向量: $(full_src)" * (TL_FULL_SRC == "auto" ? " (自动选择)" : "") *
                (full_src == "hybrid" ? "" :
                 full_src == "m52_hyb" ? " —— 其它实例独立优化(无迁移)的结果, 对目标无信息泄漏" :
                 full_src == "tl_hyb" ? " —— ⚠ 有二阶泄漏(源的 tl_hyb 用过目标实例的源数据), 仅供参考" :
                 " —— 注意: 逐层链的向量质量较差, tl_full 的结果会偏低"))
    end

    # ── 阶段 2: 目标运行(LOO) ──  tl_full 放在最后一批(它可能要用 tl_hyb 的结果作源)
    TGT = Any[]; P2 = NamedTuple[]; FULL_TODO = NamedTuple[]
    ANN_NSRC = Int[]                                         # tl_ann: 每个目标可用的小 γ 分支源实例数
    pptag = (TL_SWEEP_PP > 0 || TL_POLISH_PP > 0) ? "_PP$(TL_SWEEP_PP)x$(TL_POLISH_PP)" : ""   # 深层预算随 p 增长
    tl_msuf(m) = tl_msuf0(m) * (m in TL_HYB_LIKE ? pptag : "") * (m in TL_INTERP_M ? "_I" : "")
    tl_msuf0(m) = m == :tl_full ? "_F$(TL_FULL_B)$(full_src == "hybrid" ? "" : "_src" * full_src)" :
                 m in TL_HYB_LIKE ? "_S$(TL_SWEEP_B)x$(TL_NSWEEP)P$(TL_POLISH_B)" *
                                    (TL_MS_K > 0 ? "_M$(TL_MS_K)x$(TL_MS_B)p$(TL_MS_P)" : "") *
                                    (m in (:tl_portp, :tl_portg) ? "_H$(TL_PORT_H)" : "") *
                                    (m == :tl_hyb3 ? "_X$(TL_MC_M)" : "") *
                                    (m == :tl_ann ? "_G$(TL_ANN_GMAX)" : "") *
                                    (m == :tl_annx ? "_GA$(TL_ANN_GMAX)" : "") : ""
    tl_fname(inst, c, m, rep) = "t_i$(inst)_$(c)_$(m)$(tl_msuf(m))_r$(rep).jls"
    tl_seed(inst, c, rep) = 400000 + 1000*inst + 100*rep + tl_coff(c)
    for g in GRAPHS, rep in 1:n_rep
        others = [h.inst for h in GRAPHS if h.inst != g.inst]
        rng = Random.MersenneTwister(7919*g.inst + 104729*rep)
        chosen = sort(Random.shuffle(rng, others)[1:K_src])         # 同一 (实例, 重复) 所有方法共用
        for c in TL_CONDS, m in methods
            (m == :tl_hybx && c != "gnn_warmstart") && continue       # 跨条件对照只对 GNN 预热目标有意义
            m in TL_VIRT_M && continue                                # 虚拟方法: 阶段 2 之后由组成链合成
            f = joinpath(tl_dir, tl_fname(g.inst, c, m, rep))
            r = isfile(f) ? (try deserialize(f) catch; nothing end) : nothing
            if r !== nothing
                push!(TGT, r)
            elseif m == :tl_full
                push!(FULL_TODO, (g=g, cond=c, rep=rep, chosen=chosen))
            else
                if m in TL_PORT_METHODS
                    wv = tl_wvec(g.inst, chosen, m)
                    srcs = [(κ=tl_kappa_of(SRC[(s, c)], TL_RESCALE), layers=SRC[(s, c)].layers,
                             θ=SRCGP[(s, c)], w=wv[i]) for (i, s) in enumerate(chosen)]
                elseif m == :tl_ann
                    # 分支匹配的迁移: 源 = 其它实例的"退火分支"源链(第 1 层 γ ≤ TL_ANN_GMAX), 全部 K 个都可用
                    push!(ANN_NSRC, length(chosen))
                    srcs = [(κ=tl_kappa_of(SRCA[(s, c)], TL_RESCALE), layers=SRCA[(s, c)].layers) for s in chosen]
                elseif m == :tl_annx
                    srcs = [(κ=tl_kappa_of(SRCAX[(s, c)], TL_RESCALE), layers=SRCAX[(s, c)].layers) for s in chosen]
                else
                    cs = m == :tl_hybx ? "cold_start" : c            # tl_hybx: GNN 预热目标 + 冷启动的源数据库
                    srcs = [(κ=tl_kappa_of(SRC[(s, cs)], TL_RESCALE), layers=SRC[(s, cs)].layers) for s in chosen]
                end
                xm = m == :tl_hyb3 ? TL_MC_M : 1                         # tl_hyb3: 所有预算 ×M
                push!(P2, (g=g, cond=c, method=m, rep=rep, p_max=p_max, budget=xm*B_tgt, srcs=srcs,
                           srcx=NamedTuple[], full_budget=TL_FULL_B, sweep=xm*TL_SWEEP_B, nsweep=TL_NSWEEP,
                           polish=xm*TL_POLISH_B, ms_k=TL_MS_K, ms_b=xm*TL_MS_B, ms_p=TL_MS_P, ann_gmax=TL_ANN_GMAX,
                           sweep_pp=xm*TL_SWEEP_PP, polish_pp=xm*TL_POLISH_PP, interp=(m in TL_INTERP_M),
                           fname=tl_fname(g.inst, c, m, rep), src_insts=chosen,
                           seed=tl_seed(g.inst, c, rep) + (m == :tl_hybs2 ? 50000 : m == :tl_hybs3 ? 60000 : 0)))
            end
        end
    end
    if !isempty(SRCA)
        for c in TL_CONDS
            g1n = count(g -> SRC[(g.inst, c)].layers[1].ubest[1] <= TL_ANN_GMAX, GRAPHS)
            pq = min(5, p_max)
            ra1 = [(-SRC[(g.inst, c)].layers[1].ybest, -SRCA[(g.inst, c)].layers[1].ybest) for g in GRAPHS]
            ra5 = [(-SRC[(g.inst, c)].layers[pq].ybest, -SRCA[(g.inst, c)].layers[pq].ybest) for g in GRAPHS]
            raL = [(-SRC[(g.inst, c)].layers[p_max].ybest, -SRCA[(g.inst, c)].layers[p_max].ybest) for g in GRAPHS]
            @printf("tl_ann [%s]: 普通源链第 1 层落在小 γ 分支 %d/%d; 源链平均 R_a 普通/退火分支: p=1 %.4f/%.4f, p=%d %.4f/%.4f, p=%d %.4f/%.4f\n",
                    c, g1n, length(GRAPHS), mean(first.(ra1)), mean(last.(ra1)), pq, mean(first.(ra5)), mean(last.(ra5)),
                    p_max, mean(first.(raL)), mean(last.(raL)))
        end
    end
    if !isempty(SRCAX)
        for c in TL_CONDS
            pq = min(5, p_max)
            vx = L -> mean(-SRCAX[(g.inst, c)].layers[L].ybest for g in GRAPHS)
            @printf("tl_annx [%s]: 全层退火源链平均 R_a: p=1 %.4f, p=%d %.4f, p=%d %.4f\n", c, vx(1), pq, vx(pq), p_max, vx(p_max))
        end
    end
    @printf("阶段 2(目标运行): 共 %d 个, 已完成 %d, 本次计算 %d (+ tl_full %d)\n",
            length(TGT) + length(P2) + length(FULL_TODO), length(TGT), length(P2), length(FULL_TODO))
    sort!(P2, by = tk -> tk.method in (:tl_mtgp, :tl_hyb3) ? 0 : (tk.method == :tl_mean || tk.method in TL_HYB_LIKE ? 1 : 2))   # 慢的先派
    t_p2 = time(); tdir = tl_dir
    # dir 作为参数传进来(闭包捕获局部变量, 随任务一起发给 worker), 不依赖 worker 上的全局变量
    run_tl_batch(P, dir::String) = isempty(P) ? Any[] : pmap(P) do tk
        r = tl_target_task(tk)
        try serialize(joinpath(dir, tk.fname), r) catch; end
        wlog(@sprintf("    [迁移] inst%d %-13s %-9s r%d  R_a(p=1..%d) = %s  [%.1f 分钟, 拟合 %.0f%%]",
                      tk.g.inst, tk.cond, String(tk.method), tk.rep, tk.p_max,
                      join((@sprintf("%.4f", -L.ybest) for L in r.layers), " "),
                      r.t_total/60, 100*r.t_fit/max(r.t_total, 1e-9)), :always)
        r
    end
    append!(TGT, run_tl_batch(P2, tdir))
    if !isempty(FULL_TODO)
        # 源向量: 第 s 个源实例在深度 p 的整向量(同条件; tl_hyb 源还要求同一重复)
        function full_src_vec(s, c, rep, p)
            if full_src == "hybrid"
                return get(BASEX, (s, c, p), nothing)
            elseif full_src in ("tl_hyb", "m52_hyb")
                msrc = Symbol(full_src)
                j = findfirst(x -> x.inst == s && x.cond == c && x.method == msrc && x.rep == rep, TGT)
                return j === nothing ? nothing : TGT[j].layers[p].xfull
            else
                return vcat([SRC[(s, c)].layers[q].ubest for q in 1:p]...)
            end
        end
        P2F = NamedTuple[]
        for ft in FULL_TODO
            srcx = [(κ=tl_kappa_of(SRC[(s, ft.cond)], TL_RESCALE),
                     x=Dict(p => v for (p, v) in ((p, full_src_vec(s, ft.cond, ft.rep, p)) for p in 1:p_max) if v !== nothing))
                    for s in ft.chosen]
            push!(P2F, (g=ft.g, cond=ft.cond, method=:tl_full, rep=ft.rep, p_max=p_max, budget=B_tgt,
                        srcs=NamedTuple[], srcx=srcx, full_budget=TL_FULL_B, sweep=TL_SWEEP_B, nsweep=TL_NSWEEP,
                        polish=TL_POLISH_B, ms_k=TL_MS_K, ms_b=TL_MS_B, ms_p=TL_MS_P,
                        fname=tl_fname(ft.g.inst, ft.cond, :tl_full, ft.rep),
                        src_insts=ft.chosen, seed=tl_seed(ft.g.inst, ft.cond, ft.rep)))
        end
        @printf("  tl_full: %d 个任务(源 = %s)\n", length(P2F), full_src)
        append!(TGT, run_tl_batch(P2F, tdir))
    end
    @printf("阶段 2 完成, 墙钟 %.1f 分钟\n", (time()-t_p2)/60)

    # ── 虚拟方法 tl_mc / tl_mcs / tl_mcm: 由组成链合成(不花新演化) ──
    for vm in TL_VIRT_M
        vm in methods || continue
        vch = TL_VIRT[vm]
        n_mc = Ref(0)                          # Ref: 顶层 for 循环里不能直接给全局变量 += (软作用域)
        for g in GRAPHS, c in TL_CONDS, rep in 1:n_rep
            ch = [findfirst(x -> x.inst == g.inst && x.cond == c && x.method == m && x.rep == rep, TGT) for m in vch]
            any(isnothing, ch) && continue
            rs = [TGT[j] for j in ch]
            layers = NamedTuple[]
            for p in 1:p_max
                ys = [r.layers[p].ybest for r in rs]; w = argmin(ys)
                Lw = rs[w].layers[p]
                push!(layers, (p=p, curve=[ys[w]], ybest=ys[w], ubest=Lw.ubest, xfull=Lw.xfull,
                               nev=sum(r.layers[p].nev for r in rs),
                               diag=Dict{Symbol,Float64}(:win => w), y_zs=NaN, y0=NaN))
            end
            push!(TGT, (inst=g.inst, cond=c, method=vm, rep=rep, layers=layers, src_insts=rs[1].src_insts,
                        t_total=sum(r.t_total for r in rs), t_fit=sum(r.t_fit for r in rs)))
            n_mc[] += 1
        end
        println("$(vm): 合成 $(n_mc[]) 条(组成链 $(join(vch, " + ")))")
    end

    # ── 标准答案: p=1 网格; p≥2 = max(基准, 源数据库, 所有目标运行) ──
    REF = Dict{Tuple{Int,String,Int},Float64}()
    for g in GRAPHS, c in TL_CONDS, p in 1:p_max
        vals = vcat([-SRC[(g.inst, c)].layers[p].ybest],
                    [-r.layers[p].ybest for r in TGT if r.inst == g.inst && r.cond == c])
        # Ωt≠2 时 hybrid 基准是另一个哈密顿量下的值, 不能当本曲面的标准答案(表 T3/T4 里仍列出它作参照)
        (BASE_SAME_OM && (c != "gnn_warmstart" || BASE_SAME_WS) && haskey(BASE, (g.inst, c, p))) && push!(vals, BASE[(g.inst, c, p)])
        p == 1 && haskey(GRID, (g.inst, c)) && push!(vals, GRID[(g.inst, c)])
        REF[(g.inst, c, p)] = maximum(vals)
    end
    gap_k(r, p, k)  = REF[(r.inst, r.cond, p)] + r.layers[p].curve[min(k, length(r.layers[p].curve))]
    gap_fin(r, p)   = REF[(r.inst, r.cond, p)] + r.layers[p].ybest
    se_(v) = length(v) > 1 ? std(v)/sqrt(length(v)) : 0.0
    sign_p(w, l) = (n = w + l; n == 0 ? 1.0 :
                    min(1.0, 2*sum(Float64(binomial(BigInt(n), i)) for i in 0:min(w, l)) / 2.0^n))
    n_beat = count(r -> -r.layers[1].ybest > get(GRID, (r.inst, r.cond), Inf) + 1e-6, TGT)
    println("p=1 网格核查: 有 $(n_beat) 次目标运行超过了网格+精修的值" *
            (n_beat == 0 ? " (标准答案可信)" : " —— 网格分辨率可能不够, 已自动取最好值"))

    # ── 表 R: 参数集中与重标度(只用源数据库, 不花新演化) ──
    mprintln("\n" * "="^78)
    mprintln("【表R】参数集中 + γ 重标度检验 (源数据库, 每个实例自己的逐层最优 u*_p)")
    mprintln("      LOO 迁移误差 = |γ*_t − median_{s≠t}(γ*_s·κ_s/κ_t)| / γ 量程;  β 不重标度")
    mprintln("      '集中' = 误差 < 0.05 的实例比例。比较三种 κ: none / mean J (Shaydulin) / max J (Montañez-Barrera)")
    mprintln("="^78)
    κcv = std([SRC[(g.inst, TL_CONDS[1])].kmean for g in GRAPHS]) / mean([SRC[(g.inst, TL_CONDS[1])].kmean for g in GRAPHS])
    mprintln(@sprintf("  实例间 mean J 的变异系数 = %.3f  (越大, 重标度越重要)", κcv))
    for c in TL_CONDS, p in 1:p_max
        cells = String[]
        for mode in ("none", "mean", "max")
            errs = Float64[]
            for g in GRAPHS
                κt = tl_kappa_of(SRC[(g.inst, c)], mode)
                pred = median([SRC[(h.inst, c)].layers[p].ubest[1] * tl_kappa_of(SRC[(h.inst, c)], mode) / κt
                               for h in GRAPHS if h.inst != g.inst])
                push!(errs, abs(SRC[(g.inst, c)].layers[p].ubest[1] - pred) / BOX_HI[1])
            end
            push!(cells, @sprintf("γ[%s] 中位误差 %.3f 集中 %2d/%d", mode, median(errs), count(<(0.05), errs), length(errs)))
        end
        berr = [abs(SRC[(g.inst, c)].layers[p].ubest[2] -
                    median([SRC[(h.inst, c)].layers[p].ubest[2] for h in GRAPHS if h.inst != g.inst])) / BOX_HI[2]
                for g in GRAPHS]
        mprintln(@sprintf("  p=%d %-13s ", p, c) * join(cells, " | ") *
                 @sprintf(" | β 中位误差 %.3f 集中 %2d/%d", median(berr), count(<(0.05), berr), length(berr)))
    end

    # ── 表 T1: 主表 ──
    ks = sort(unique(filter(k -> k <= B_tgt, [1, 3, 5, 10, 15, 20, 30, 50, B_tgt])))
    tol_s = 1e-3
    mprintln("\n" * "="^78)
    mprintln("【表T1】逐层链: 第 p 层用完 k 次演化后与标准答案的差距 (R_a 单位, 均值±标准误; 越小越好)")
    mprintln("      每格 = 实例 × 重复; p≥2 的差距包含前面各层留下的前缀质量(端到端指标)")
    mprintln("      '零样本' = 第 p 层只评估迁移中位数点(第 1 次新演化)时的差距 —— 对应 Shaydulin 的直接迁移")
    mprintln("      '达标' = 第 p 层结束时差距 < $(tol_s); 每条链到第 p 层共用 p×$(B_tgt) 次演化")
    mprintln("="^78)
    SUM = NamedTuple[]
    for p in 1:p_max, c in TL_CONDS
        mprintln("\n  p=$(p)  $(c)" * (p == 1 ? "  (标准答案 = 网格)" :
                 "  (标准答案 = 源数据库/所有运行" * (isempty(BASE) ? "" : "/基准") * " 的最好值)"))
        mprintln("  " * rpad("方法", 10) * join([lpad("@$(k)", 17) for k in ks]) * "      零样本   达标  分钟/链")
        for m in methods
            (m in TL_HYB_LIKE || m == :tl_full || m in TL_VIRT_M) && continue    # 这几种的预算口径不同, 见表 T3
            rs = [r for r in TGT if r.cond == c && r.method == m]
            isempty(rs) && continue
            cells = [lpad(@sprintf("%.4f±%.4f", mean([gap_k(r, p, k) for r in rs]), se_([gap_k(r, p, k) for r in rs])), 17)
                     for k in ks]
            zs = [REF[(r.inst, r.cond, p)] + r.layers[p].y_zs for r in rs]
            zs = filter(isfinite, zs)
            fin = [gap_fin(r, p) for r in rs]
            mprintln("  " * rpad(String(m), 10) * join(cells) *
                     @sprintf("   %8s   %2d/%-2d  %6.2f", isempty(zs) ? "-" : @sprintf("%.4f", mean(zs)),
                              count(<(tol_s), fin), length(fin), mean(r.t_total for r in rs)/60))
            push!(SUM, (p=p, cond=c, method=m, n=length(rs), gaps=[mean([gap_k(r, p, k) for r in rs]) for k in ks],
                        zero_shot=isempty(zs) ? NaN : mean(zs), solved=count(<(tol_s), fin)))
        end
        sg = [REF[(g.inst, c, p)] + SRC[(g.inst, c)].layers[p].ybest for g in GRAPHS]
        mprintln(@sprintf("  %-10s 第 p 层结束差距 %.4f±%.4f  (每层 %d 次 BO + %d 次精修, 达标 %d/%d)",
                          "[源数据库]", mean(sg), se_(sg), B_src, TL_SRC_POLISH, count(<(tol_s), sg), length(sg)))
        if !isempty(BASE)
            bg = [REF[(g.inst, c, p)] - BASE[(g.inst, c, p)] for g in GRAPHS if haskey(BASE, (g.inst, c, p))]
            isempty(bg) || mprintln(@sprintf("  %-10s 差距 %.4f±%.4f  (束搜索 + 轮扫 + 精修, 每层约数百~上千次演化)",
                                             "[hybrid]", mean(bg), se_(bg)))
        end
    end

    # ── 表 T2: 配对比较 ──
    mprintln("\n" * "="^78)
    mprintln("【表T2】配对比较 (同实例/同条件/同重复 ⇒ 同一组源实例)")
    mprintln("      Δ = 对手差距 − 本方法差距 (>0 表示本方法更好); |Δ|<1e-4 记持平")
    mprintln("="^78)
    # 所有迁移方法 vs matern52(迁移有没有用); GP 迁移方法 vs tl_nm(GP 比 Shaydulin 式局部精修强不强)
    cmp_pairs = vcat([(m, :matern52) for m in methods if !(m == :matern52 || m == :tl_full || m in TL_VIRT_M || m in TL_HYB_LIKE) && :matern52 in methods],
                     [(m, :tl_nm) for m in methods if m in (:tl_init, :tl_mean, :tl_mtgp) && :tl_nm in methods])
    for p in 1:p_max, c in TL_CONDS, k in unique([min(10, B_tgt), B_tgt]), (m, o) in cmp_pairs
        d = Float64[]
        for r in TGT
            (r.cond == c && r.method == m) || continue
            j = findfirst(x -> x.cond == c && x.method == o && x.inst == r.inst && x.rep == r.rep, TGT)
            j === nothing && continue
            push!(d, gap_k(TGT[j], p, k) - gap_k(r, p, k))
        end
        isempty(d) && continue
        w = count(>(1e-4), d); l = count(<(-1e-4), d)
        mprintln(@sprintf("  p=%d %-13s @%-3d %-9s vs %-9s Δ=%+.5f ± %.5f   赢/平/输 %2d/%2d/%2d   p_sign=%.3g",
                          p, c, k, String(m), String(o), mean(d), se_(d), w, length(d)-w-l, l, sign_p(w, l)))
    end

    # ── 表 D: 迁移模型诊断 ──
    if any(m -> m in (:tl_mean, :tl_mtgp), methods)
        mprintln("\n" * "="^78)
        mprintln("【表D】迁移模型学到了什么 (最后一次拟合的超参数, 实例×重复 的均值)")
        mprintln("      tl_mtgp: ρ = 源-目标相关系数 (→1 源可信, →0 自动退化为单任务); a = 目标/源 幅度比")
        mprintln("      tl_mean: δ幅度 = 实例特有残差的标准差 / 源增益标准差 (越小说明群体曲面越能直接搬用)")
        mprintln("="^78)
        for p in 1:p_max, c in TL_CONDS
            line = @sprintf("  p=%d %-13s", p, c)
            rm = [r for r in TGT if r.cond == c && r.method == :tl_mtgp && haskey(r.layers[p].diag, :rho)]
            isempty(rm) || (line *= @sprintf("  tl_mtgp: ρ=%.3f±%.3f  a=%.3f  σ_S²=%.2e",
                                             mean(r.layers[p].diag[:rho] for r in rm), se_([r.layers[p].diag[:rho] for r in rm]),
                                             mean(r.layers[p].diag[:a] for r in rm), mean(r.layers[p].diag[:snS] for r in rm)))
            rb = [r for r in TGT if r.cond == c && r.method == :tl_mean && haskey(r.layers[p].diag, :dsd)]
            isempty(rb) || (line *= @sprintf("   tl_mean: δ幅度=%.3f±%.3f", mean(r.layers[p].diag[:dsd] for r in rb),
                                             se_([r.layers[p].diag[:dsd] for r in rb])))
            mprintln(line)
        end
    end

    # ── 表 T3: 达到 hybrid 质量要多少次演化 ──
    if !isempty(BASE) && any(m -> m in TL_HYB_LIKE || m == :tl_full || m in TL_VIRT_M, methods)
        cum_ev(r, p) = r.method == :tl_full ? r.layers[p].nev : sum(r.layers[q].nev for q in 1:p)
        hgap(r, p, k) = BASE[(r.inst, r.cond, p)] + r.layers[p].curve[min(k, length(r.layers[p].curve))]
        hgap_fin(r, p) = BASE[(r.inst, r.cond, p)] + r.layers[p].ybest
        mprintln("\n" * "="^78)
        mprintln("【表T3】与 hybrid 基准对比: 深度 p 的解要花多少次演化、差多少")
        mprintln("      差距 = hybrid 最优 R_a − 本方法 R_a (>0 表示不如 hybrid); '追平' = 差距 < 1e-3; '超过' = 差距 < −1e-4")
        mprintln("      hybrid 若有多个种子则取最好(种子数见各段第一行); m52_hyb = tl_hyb 去掉迁移(新层用纯 Matérn), 隔离迁移的贡献")
        mprintln("      累计演化: 链式方法 = 第 1..p 层之和; tl_full = 该深度自身(不需要前面的层); hybrid = 单个种子跑到深度 p 之和")
        mprintln("="^78)
        for p in 1:p_max, c in TL_CONDS
            any(g -> haskey(BASE, (g.inst, c, p)), GRAPHS) || continue      # p>5: 没有 hybrid, 见表 Deep
            mprintln("\n  p=$(p)  $(c)")
            hn = vcat([get(BASEN, (g.inst, c, p), Int[]) for g in GRAPHS]...)
            nseed = isempty(hn) ? 0 : round(Int, length(hn) / count(g -> haskey(BASEN, (g.inst, c, p)), GRAPHS))
            mprintln(@sprintf("  %-10s 累计演化 %9s   (每实例 %d 个种子, 取最好者作为基准)", "hybrid",
                              isempty(hn) ? "未知" : @sprintf("%.0f", mean(hn)), nseed))
            for m in (:tl_mcx, :tl_mca, :tl_mc, :tl_mcs, :tl_mcm, :tl_hyb3, :tl_full, :tl_hyb, :tl_annx, :tl_ann, :tl_port, :tl_portp, :tl_portg, :tl_hybx, :m52_hyb, :tl_init, :matern52)
                m in methods || continue
                rs = [r for r in TGT if r.cond == c && r.method == m && haskey(BASE, (r.inst, r.cond, p))]
                isempty(rs) && continue
                d = [hgap_fin(r, p) for r in rs]; ev = [cum_ev(r, p) for r in rs]
                zs = filter(isfinite, [BASE[(r.inst, r.cond, p)] + r.layers[p].y_zs for r in rs])
                line = @sprintf("  %-10s 累计演化 %9.0f   差距 %+.4f±%.4f   追平 %2d/%-2d  超过 %2d", String(m), mean(ev),
                                mean(d), se_(d), count(<(1e-3), d), length(d), count(<(-1e-4), d))
                ((m == :tl_full || (m in TL_HYB_LIKE && !(m in (:m52_hyb, :tl_hyb3, :tl_hybs2, :tl_hybs3)))) && !isempty(zs)) &&
                    (line *= @sprintf("   零样本差距 %+.4f", mean(zs)))
                if !isempty(hn)
                    line *= @sprintf("   演化数 = hybrid 的 %.1f%%", 100*mean(ev)/mean(hn))
                end
                mprintln(line)
            end
            rf = [r for r in TGT if r.cond == c && r.method == :tl_full && haskey(BASE, (r.inst, r.cond, p))]
            if !isempty(rf)
                kf = sort(unique(filter(k -> k <= TL_FULL_B, [1, 5, 10, 25, 50, 100, 150, 200, 300, TL_FULL_B])))
                mprintln("  tl_full 收敛: " * join([@sprintf("@%d %+.4f(追平 %d)", k, mean(hgap(r, p, k) for r in rf),
                                                           count(r -> hgap(r, p, k) < 1e-3, rf)) for k in kf], "  "))
            end
            for (ma, mb) in ((:tl_mcx, :tl_mca), (:tl_annx, :tl_ann), (:tl_mca, :tl_mc), (:tl_ann, :tl_hyb), (:tl_mc, :tl_hyb3), (:tl_mc, :tl_mcs), (:tl_mcs, :tl_hyb3), (:tl_mcm, :tl_mc),
                             (:tl_mc, :tl_hyb), (:tl_hyb3, :tl_hyb),
                             (:tl_hyb, :m52_hyb), (:tl_full, :tl_hyb), (:tl_full, :m52_hyb),
                             (:tl_port, :tl_hyb), (:tl_portp, :tl_hyb), (:tl_portg, :tl_hyb),
                             (:tl_portp, :tl_port), (:tl_portg, :tl_port), (:tl_port, :m52_hyb),
                             (:tl_hyb, :tl_hybx))
                (ma in methods && mb in methods) || continue
                dd = Float64[]
                for r in TGT
                    (r.cond == c && r.method == ma) || continue
                    j = findfirst(x -> x.cond == c && x.method == mb && x.inst == r.inst && x.rep == r.rep, TGT)
                    j === nothing || push!(dd, TGT[j].layers[p].ybest - r.layers[p].ybest)     # >0: ma 更好
                end
                isempty(dd) && continue
                ww = count(>(1e-4), dd); ll = count(<(-1e-4), dd)
                mprintln(@sprintf("  配对 %-8s vs %-8s  ΔR_a=%+.4f±%.4f  赢/平/输 %2d/%2d/%2d  p_sign=%.3g",
                                  String(ma), String(mb), mean(dd), se_(dd), ww, length(dd)-ww-ll, ll, sign_p(ww, ll)))
            end
            for mm in TL_HYB_LIKE
                mm in methods || continue
                rms = [r for r in TGT if r.cond == c && r.method == mm && haskey(r.layers[p].diag, :gain_ms)]
                isempty(rms) && continue
                mprintln(@sprintf("  %s 新层多起点: 平均改进 %.4f, 换到别的盆地 %d/%d 次", String(mm),
                                  mean(r.layers[p].diag[:gain_ms] for r in rms),
                                  count(r -> r.layers[p].diag[:ms_switched] > 0.5, rms), length(rms)))
            end
            rh = [r for r in TGT if r.cond == c && r.method == :tl_hyb && haskey(r.layers[p].diag, :gain_sweep)]
            if !isempty(rh) && p >= 2
                mprintln(@sprintf("  tl_hyb 各阶段平均改进: 轮扫 %.4f, 全维精修 %.4f (R_a 单位)",
                                  mean(r.layers[p].diag[:gain_sweep] for r in rh) / 1.0,
                                  mean(r.layers[p].diag[:gain_polish] for r in rh) / 1.0))
            end
        end
    end

    # ── 表 MC: 多链迁移 vs 等预算单链 —— 多样性是否比预算更值钱 ──
    if any(m -> m in TL_VIRT_M, methods)
        bref(r, p) = haskey(BASE, (r.inst, r.cond, p)) ? BASE[(r.inst, r.cond, p)] : REF[(r.inst, r.cond, p)]
        mprintln("\n" * "="^78)
        mprintln("【表MC】多链迁移: $(join(TL_MC_CH, " + ")) 各一条链, 每个深度取最好 (差距相对 " *
                 (isempty(BASE) ? "标准答案" : "hybrid") * "; >0 表示更差)")
        mprintln("      对照 tl_hyb3 = 一条 tl_hyb 链但所有预算 ×$(TL_MC_M) —— 总演化数与 tl_mc 相当;")
        mprintln("      若 tl_mc 明显好于 tl_hyb3 ⇒ 开局多样性(而不是预算)在起作用。")
        mprintln("      '种子取优' = 同一方法两个重复取最好(多样性只来自随机种子, 不来自迁移开局), 仅供参考")
        mprintln("      '2链' = 只用前两条组成链; '胜出' = 每条组成链在该深度严格最好(>1e-4)的次数")
        mprintln("="^78)
        for p in 1:p_max, c in TL_CONDS
            mprintln("\n  p=$(p)  $(c)")
            for m in (:m52_hyb, :tl_hyb, :tl_hyb3, :tl_mcs, :tl_mc, :tl_mcm)
                m in methods || continue
                rs = [r for r in TGT if r.cond == c && r.method == m]
                isempty(rs) && continue
                d = [bref(r, p) + r.layers[p].ybest for r in rs]
                ev = [sum(r.layers[q].nev for q in 1:p) for r in rs]
                mprintln(@sprintf("  %-8s 差距 %+.4f±%.4f  中位数 %+.4f  追平 %2d/%-2d  累计演化 %7.0f", String(m),
                                  mean(d), se_(d), median(d), count(<(1e-3), d), length(d), mean(ev)))
            end
            # 2 链(前两条组成链)与"同法两种子取优"
            if :tl_mc in methods && length(TL_MC_CH) >= 2
                d2 = Float64[]
                for r in TGT
                    (r.cond == c && r.method == TL_MC_CH[1]) || continue
                    j = findfirst(x -> x.cond == c && x.method == TL_MC_CH[2] && x.inst == r.inst && x.rep == r.rep, TGT)
                    j === nothing || push!(d2, bref(r, p) + min(r.layers[p].ybest, TGT[j].layers[p].ybest))
                end
                ds = Float64[]
                for g in GRAPHS
                    rr = [r for r in TGT if r.cond == c && r.method == TL_MC_CH[1] && r.inst == g.inst]
                    length(rr) >= 2 && push!(ds, bref(rr[1], p) + minimum(r.layers[p].ybest for r in rr[1:2]))
                end
                isempty(d2) || mprintln(@sprintf("  2链 %s+%s 取优 差距 %+.4f   |   %s 两种子取优 差距 %s", String(TL_MC_CH[1]),
                                                 String(TL_MC_CH[2]), mean(d2), String(TL_MC_CH[1]),
                                                 isempty(ds) ? "-" : @sprintf("%+.4f", mean(ds))))
            end
            rm = [r for r in TGT if r.cond == c && r.method == :tl_mc]
            if !isempty(rm)
                wins = zeros(Int, length(TL_MC_CH))
                for r in rm
                    j0 = [findfirst(x -> x.inst == r.inst && x.cond == c && x.method == m && x.rep == r.rep, TGT) for m in TL_MC_CH]
                    ys = [TGT[j].layers[p].ybest for j in j0]; w = argmin(ys)
                    all(k -> k == w || ys[k] > ys[w] + 1e-4, eachindex(ys)) && (wins[w] += 1)
                end
                mprintln("  胜出: " * join(("$(TL_MC_CH[k]) $(wins[k])" for k in eachindex(wins)), ", ") *
                         "  (其余为并列, 共 $(length(rm)) 条)")
            end
        end
    end

    # ── 表 F: 2×2 析因 —— {冷启动, GNN 预热} × {无迁移 m52_hyb, 迁移 tl_hyb / tl_mc} ──
    if :m52_hyb in methods && :tl_hyb in methods && length(TL_CONDS) == 2
        ra_f(i, c, m, p) = (v = [-r.layers[p].ybest for r in TGT if r.inst == i && r.cond == c && r.method == m];
                            isempty(v) ? NaN : mean(v))
        mprintln("\n" * "="^78)
        mprintln("【表F】2×2 析因: GNN 预热 × 迁移 (R_a 绝对值, 每实例先对重复取平均, 再对实例平均±标准误)")
        mprintln("      迁移效应 = 迁移 − 无迁移(同条件); GNN 效应 = GNN 预热 − 冷启动(同优化器)")
        mprintln("      交互 = 迁移效应(GNN) − 迁移效应(冷启动): >0 两者协同, ≈0 简单叠加, <0 互相替代")
        mprintln("="^78)
        for p in 1:p_max, tm in (:tl_hyb, :tl_mc)
            tm in methods || continue
            ins = [g.inst for g in GRAPHS if all(isfinite, (ra_f(g.inst, "cold_start", :m52_hyb, p), ra_f(g.inst, "cold_start", tm, p),
                                                            ra_f(g.inst, "gnn_warmstart", :m52_hyb, p), ra_f(g.inst, "gnn_warmstart", tm, p)))]
            isempty(ins) && continue
            Mc = [ra_f(i, "cold_start", :m52_hyb, p) for i in ins];    Tc = [ra_f(i, "cold_start", tm, p) for i in ins]
            Mg = [ra_f(i, "gnn_warmstart", :m52_hyb, p) for i in ins]; Tg = [ra_f(i, "gnn_warmstart", tm, p) for i in ins]
            fx(v) = @sprintf("%+.4f±%.4f", mean(v), se_(v))
            dI = (Tg .- Mg) .- (Tc .- Mc)
            wI = count(>(1e-4), dI); lI = count(<(-1e-4), dI)
            mprintln(@sprintf("  p=%d 迁移=%-7s n=%2d | 冷/无 %.4f  冷/迁 %.4f  GNN/无 %.4f  GNN/迁 %.4f", p, String(tm), length(ins),
                              mean(Mc), mean(Tc), mean(Mg), mean(Tg)))
            mprintln("        迁移效应: 冷 " * fx(Tc .- Mc) * "  GNN " * fx(Tg .- Mg) *
                     " | GNN 效应: 无迁移 " * fx(Mg .- Mc) * "  有迁移 " * fx(Tg .- Tc) *
                     " | 交互 " * fx(dI) * @sprintf("  (+/−: %d/%d, p_sign=%.3g)", wI, lI, sign_p(wI, lI)))
        end
    end

    # ── 表 P: 开局诊断 —— 5 个迁移开局点本身好不好(在任何 BO / 精修之前) ──
    OPEN_M = [m for m in (:tl_hyb, :tl_port, :tl_portp, :tl_portg, :tl_hybx) if m in methods]
    if length(OPEN_M) >= 2 || any(m -> m in TL_PORT_METHODS, OPEN_M)
        # 开局之后第 k0 次演化: p=1 还要先付原点 1 次; p≥2 原点免费。缓存的旧结果没有 :n_pts 时按 5 个算
        k_open(r, p) = (p == 1 ? 1 : 0) + round(Int, get(r.layers[p].diag, :n_pts, Float64(TL_N_TRANSFER)))
        mprintln("\n" * "="^78)
        mprintln("【表P】开局诊断: 只评估完原点 + 迁移开局点之后, 与标准答案的差距(第 p 层; 越小越好)")
        mprintln("      '命中' = 开局后差距 < 0.005 (开局点已落在正确的盆地里, 后面的 BO/精修只需细调)")
        mprintln("      预测遗憾 = 组合在源实例分布上的加权期望遗憾(源 GP 预测, R_a 单位); 括号里 = 同一目标函数下")
        mprintln("               中位数+KDE 开局点的值。覆盖 = 遗憾 < 1e-3 的源权重占比。ESS = 源权重的有效样本数")
        mprintln("="^78)
        for p in 1:p_max, c in TL_CONDS
            mprintln("\n  p=$(p)  $(c)")
            for m in OPEN_M
                rs = [r for r in TGT if r.cond == c && r.method == m]
                isempty(rs) && continue
                go = [gap_k(r, p, k_open(r, p)) for r in rs]
                gz = filter(isfinite, [REF[(r.inst, r.cond, p)] + r.layers[p].y_zs for r in rs])
                line = @sprintf("  %-9s 开局后差距 %.4f±%.4f  命中 %2d/%-2d  第1点差距 %s  最终 %.4f",
                                String(m), mean(go), se_(go), count(<(0.005), go), length(go),
                                isempty(gz) ? "   -  " : @sprintf("%.4f", mean(gz)), mean(gap_fin(r, p) for r in rs))
                rp = [r for r in rs if haskey(r.layers[p].diag, :port_er)]
                if !isempty(rp)
                    dgs = [r.layers[p].diag for r in rp]
                    line *= @sprintf("   预测遗憾 %.4f (%.4f)  覆盖 %.2f (%.2f)  ESS %.1f",
                                     mean(d[:port_er] for d in dgs), mean(d[:port_er_med] for d in dgs),
                                     mean(d[:port_cov] for d in dgs), mean(d[:port_cov_med] for d in dgs),
                                     mean(d[:ess] for d in dgs))
                end
                mprintln(line)
            end
            for (ma, mb) in ((:tl_port, :tl_hyb), (:tl_portp, :tl_hyb), (:tl_portg, :tl_hyb), (:tl_hyb, :tl_hybx))
                (ma in methods && mb in methods) || continue
                dd = Float64[]
                for r in TGT
                    (r.cond == c && r.method == ma) || continue
                    jj = findfirst(x -> x.cond == c && x.method == mb && x.inst == r.inst && x.rep == r.rep, TGT)
                    jj === nothing || push!(dd, gap_k(TGT[jj], p, k_open(TGT[jj], p)) - gap_k(r, p, k_open(r, p)))
                end
                isempty(dd) && continue
                ww = count(>(1e-4), dd); ll = count(<(-1e-4), dd)
                mprintln(@sprintf("  开局配对 %-8s vs %-8s  Δ差距=%+.4f±%.4f  赢/平/输 %2d/%2d/%2d  p_sign=%.3g",
                                  String(ma), String(mb), mean(dd), se_(dd), ww, length(dd)-ww-ll, ll, sign_p(ww, ll)))
            end
        end
    end

    # ── 表 T4: 逐实例 —— 剩下的差距集中在哪些实例上 ──
    T4_M = [m for m in (:tl_hyb, :tl_mc, :tl_mca, :tl_mcx, :tl_ann, :tl_annx, :tl_mcs, :tl_mcm, :tl_hyb3, :tl_port, :tl_portp, :tl_portg, :tl_hybx, :m52_hyb, :tl_init) if m in methods]
    if !isempty(BASE) && any(m -> m in TL_HYB_LIKE, T4_M)
        m_ref = first(m for m in T4_M if m in TL_HYB_LIKE)      # 离群判据用的方法(优先 tl_hyb)
        mprintln("\n" * "="^78)
        mprintln("【表T4】逐实例: 与 hybrid 的差距(各重复分开列; >0 表示不如 hybrid)")
        mprintln("      '贪心链' = 源数据库逐层链(无迁移、无回头修)与 hybrid 的差距 —— 衡量这个实例对普通方法有多难")
        mprintln("      标 ◀ 的是 $(m_ref) 平均差距 > 0.005 的离群实例; 最后几列 = 实例特征(z 分数), 看离群是否可预测")
        mprintln("="^78)
        for c in TL_CONDS, p in unique([1, min(p_max, 5), p_max])
            hasb = any(g -> haskey(BASE, (g.inst, c, p)), GRAPHS)
            mprintln("\n  p=$(p)  $(c)" * (hasb ? "" : "   (此深度没有 hybrid: 'hybrid R_a' 列换成各方法运行中的最好值)"))
            mprintln("  " * rpad("实例", 6) * lpad("hybrid R_a", 11) * lpad("贪心链", 9) * "   " *
                     join([rpad(String(m), 8*n_rep + 2) for m in T4_M]) * "  z(w_cv,unw,fid)")
            outl = Int[]
            for g in GRAPHS
                bv = get(BASE, (g.inst, c, p), REF[(g.inst, c, p)])
                line = rpad("i$(g.inst)", 6) * lpad(@sprintf("%.4f", bv), 11) *
                       lpad(@sprintf("%+.4f", bv + SRC[(g.inst, c)].layers[p].ybest), 9) * "   "
                ref_mean = NaN
                for m in T4_M
                    rs = sort([r for r in TGT if r.inst == g.inst && r.cond == c && r.method == m], by = r -> r.rep)
                    gs = [bv + r.layers[p].ybest for r in rs]
                    m == m_ref && !isempty(gs) && (ref_mean = mean(gs))
                    line *= rpad(join((@sprintf("%+.4f", x) for x in gs), " "), 8*n_rep + 2)
                end
                line *= "  " * join((@sprintf("%+.1f", z) for z in ZP[g.inst]), ",")
                if isfinite(ref_mean) && ref_mean > 0.005
                    line *= " ◀"; push!(outl, g.inst)
                end
                mprintln("  " * line)
            end
            mprintln("  离群实例: " * (isempty(outl) ? "无" : join(("i$(i)" for i in outl), ", ")))
        end
    end

    # ── 表 Deep: R_a 绝对值随深度 p 的变化(p 可到 10; 不依赖 hybrid 基准) ──
    DEEP_M = [m for m in (:m52_hyb, :tl_hyb, :tl_port, :tl_portp, :tl_ann, :tl_annx, :tl_mc, :tl_mca, :tl_mcx) if m in methods]
    if !isempty(DEEP_M)
        ra_d(m, c, p, ins) = (v = [mean(-r.layers[p].ybest for r in TGT if r.inst == i && r.cond == c && r.method == m)
                                   for i in ins if any(r -> r.inst == i && r.cond == c && r.method == m, TGT)];
                              isempty(v) ? (NaN, 0.0, 0) : (mean(v), se_(v), length(v)))
        # 冷启动的分支: hybrid 第 1 层最优 γ(没有就用源数据库链的第 1 层)
        g1of(i, c) = haskey(BASEX, (i, c, 1)) ? BASEX[(i, c, 1)][1] : SRC[(i, c)].layers[1].ubest[1]
        mprintln("\n" * "="^78)
        mprintln("【表Deep】R_a 绝对值随深度 p (每实例先对重复取平均, 再对实例平均±标准误; 'hybrid' 只到 p=5)")
        mprintln("      每种条件另按 hybrid 第 1 层 γ* 分成 小γ(≤$(TL_ANN_GMAX), 退火分支) / 大γ 两组 —— 看 tl_ann 能否把大γ组带到退火分支上")
        mprintln("="^78)
        for c in TL_CONDS
            groups = [("全部", [g.inst for g in GRAPHS])]
            push!(groups, ("小γ", [g.inst for g in GRAPHS if g1of(g.inst, c) <= TL_ANN_GMAX]))
            push!(groups, ("大γ", [g.inst for g in GRAPHS if g1of(g.inst, c) > TL_ANN_GMAX]))
            for (gname, ins) in groups
                isempty(ins) && continue
                mprintln("\n  $(c)  [$(gname), $(length(ins)) 个实例]")
                mprintln("  " * rpad("p", 4) * rpad("hybrid", 17) * join(rpad(String(m), 17) for m in DEEP_M))
                for p in 1:p_max
                    hv = [BASE[(i, c, p)] for i in ins if haskey(BASE, (i, c, p))]
                    line = rpad("$(p)", 4) * rpad(isempty(hv) ? "-" : @sprintf("%.4f±%.4f", mean(hv), se_(hv)), 17)
                    for m in DEEP_M
                        μ, se, n = ra_d(m, c, p, ins)
                        line *= rpad(n == 0 ? "-" : @sprintf("%.4f±%.4f", μ, se), 17)
                    end
                    mprintln("  " * line)
                end
                for m in (:tl_ann, :tl_annx, :tl_mc, :tl_mca, :tl_mcx)
                    m in DEEP_M || continue
                    vs = [ra_d(m, c, p, ins)[1] for p in 1:p_max]
                    all(isfinite, vs) || continue
                    mprintln("  $(m) 每层增益: " * join((@sprintf("%+.4f", vs[p] - vs[p-1]) for p in 2:p_max), " "))
                end
            end
        end
    end

    # ── 表 PP: 深层优化器改动(预算随 p 增长 / INTERP 候选) vs 对照, 同目录里的对照 checkpoint 逐实例配对 ──
    if !isempty(pptag) || !isempty(TL_INTERP_M)
        PPM = [m for m in (:tl_hyb, :tl_port, :tl_portp, :tl_ann, :tl_annx, :tl_mc, :tl_mca, :tl_mcx) if m in methods]
        # 对照链: (ra = 各深度 R_a, ne = 累计演化数); 虚拟方法由组成链合成(每深度取最好, 演化数相加); suf = 文件后缀函数
        function ld_pp0(i, c, m, rep, suf)
            if m in TL_VIRT_M
                rs = [ld_pp0(i, c, mm, rep, suf) for mm in TL_VIRT[m]]
                any(isnothing, rs) && return nothing
                return (ra=[maximum(r.ra[p] for r in rs) for p in 1:p_max], ne=[sum(r.ne[p] for r in rs) for p in 1:p_max])
            end
            f = joinpath(tl_dir, "t_i$(i)_$(c)_$(m)$(suf(m))_r$(rep).jls")
            isfile(f) || return nothing
            r = try deserialize(f) catch; nothing end
            (r === nothing || length(r.layers) < p_max) && return nothing
            return (ra=[-r.layers[p].ybest for p in 1:p_max], ne=cumsum([r.layers[p].nev for p in 1:p_max]))
        end
        function ld_pp1(i, c, m, rep)
            k = findfirst(x -> x.inst == i && x.cond == c && x.method == m && x.rep == rep, TGT)
            k === nothing && return nothing
            r = TGT[k]
            return (ra=[-r.layers[p].ybest for p in 1:p_max], ne=cumsum([r.layers[p].nev for p in 1:p_max]))
        end
        # 用到 INTERP 的方法: 虚拟方法看组成链
        uses_int(m) = m in TL_VIRT_M ? any(mm -> mm in TL_INTERP_M, TL_VIRT[m]) : (m in TL_INTERP_M)
        CMPS = Tuple{String,Function,Function}[]                 # (对照名, 后缀函数, 哪些方法适用)
        (!isempty(pptag) || !isempty(TL_INTERP_M)) && push!(CMPS, ("原设置(固定预算, 无 INTERP)", tl_msuf0, m -> true))
        (!isempty(pptag) && !isempty(TL_INTERP_M)) &&
            push!(CMPS, ("同预算、不加 INTERP", mm -> tl_msuf0(mm) * (mm in TL_HYB_LIKE ? pptag : ""), uses_int))
        mprintln("\n" * "="^78)
        mprintln("【表PP】深层优化器改动 vs 对照 (本次: " *
                 (isempty(pptag) ? "固定预算" : "每层轮扫 max($(TL_SWEEP_B), $(TL_SWEEP_PP)·p), 精修 max($(TL_POLISH_B), $(TL_POLISH_PP)·p)") *
                 (isempty(TL_INTERP_M) ? "" : "; INTERP 候选: $(join(TL_INTERP_M, ","))") * ")")
        mprintln("      同一实例/重复/种子配对; Δ = 本次 − 对照; 胜/负 = |Δ|>1e-4 的对数; 演化数为累计到该深度的平均值")
        mprintln("      改动没起作用的浅层 Δ 应 ≈ 0(确定性核查); 深层 Δ 显著 > 0 ⇒ 原来的平台期是优化器造成的")
        mprintln("="^78)
        for (cname, suf, applies) in CMPS, c in TL_CONDS, m in PPM
            applies(m) || continue
            chg = m in TL_VIRT_M ? any(mm -> suf(mm) != tl_msuf(mm), TL_VIRT[m]) : suf(m) != tl_msuf(m)
            chg || continue                                           # 本次与对照是同一批文件(这个方法没被改动)
            prs = [(ld_pp0(g.inst, c, m, rep, suf), ld_pp1(g.inst, c, m, rep)) for g in GRAPHS for rep in 1:n_rep]
            prs = [pr for pr in prs if !isnothing(pr[1]) && !isnothing(pr[2])]
            if isempty(prs)
                mprintln("\n  [对照: $(cname)] $(c) $(m): 没有可配对的 checkpoint(先用对照设置跑同样的 p_max)")
                continue
            end
            mprintln("\n  [对照: $(cname)]  $(c)  $(m)  [$(length(prs)) 对]")
            mprintln("  " * rpad("p", 4) * rpad("对照 R_a", 10) * rpad("本次 R_a", 10) * rpad("Δ±se", 18) * rpad("胜/负", 8) *
                     rpad("符号检验p", 11) * "累计演化 对照→本次")
            for p in 1:p_max
                a = [pr[1].ra[p] for pr in prs]; b = [pr[2].ra[p] for pr in prs]; d = b .- a
                w = count(>(1e-4), d); l = count(<(-1e-4), d)
                mprintln("  " * rpad("$(p)", 4) * rpad(@sprintf("%.4f", mean(a)), 10) * rpad(@sprintf("%.4f", mean(b)), 10) *
                         rpad(@sprintf("%+.4f±%.4f", mean(d), se_(d)), 18) * rpad("$(w)/$(l)", 8) *
                         rpad(@sprintf("%.3g", sign_p(w, l)), 11) *
                         @sprintf("%.0f → %.0f", mean(pr[1].ne[p] for pr in prs), mean(pr[2].ne[p] for pr in prs)))
            end
            if p_max > 5
                a5o = [pr[1].ra[5] for pr in prs]; aLo = [pr[1].ra[p_max] for pr in prs]
                a5 = [pr[2].ra[5] for pr in prs];  aL = [pr[2].ra[p_max] for pr in prs]
                mprintln(@sprintf("  p=5→%d 每层平均增益: 对照 %+.4f, 本次 %+.4f", p_max,
                                  mean(aLo .- a5o) / (p_max - 5), mean(aL .- a5) / (p_max - 5)))
            end
        end
    end

    # ── 表 I: INTERP 候选被换用的次数(它比"前缀 + BO 新层"好的实例数)与幅度 ──
    if !isempty(TL_INTERP_M)
        mprintln("\n" * "="^78)
        mprintln("【表I】INTERP 候选 vs '前缀 + BO 新层' (p≥2; 换用 = INTERP 更好, 作为轮扫/精修起点)")
        mprintln("      增益 = R_a(INTERP) − R_a(BO 新层) 的中位数; >0 表示整体拉伸比逐层追加好")
        mprintln("="^78)
        for c in TL_CONDS, m in TL_INTERP_M
            m in methods || continue
            rs = [r for r in TGT if r.cond == c && r.method == m]
            isempty(rs) && continue
            mprintln("\n  $(c)  $(m)  [$(length(rs)) 条链]")
            mprintln("  " * rpad("p", 4) * rpad("换用", 10) * rpad("增益中位数", 12) * "换用时平均增益")
            for p in 2:p_max
                gs = [get(r.layers[p].diag, :interp_gain, NaN) for r in rs]; gs = filter(isfinite, gs)
                isempty(gs) && continue
                nw = count(>(0), gs)
                mprintln("  " * rpad("$(p)", 4) * rpad("$(nw)/$(length(gs))", 10) * rpad(@sprintf("%+.4f", median(gs)), 12) *
                         (nw == 0 ? "-" : @sprintf("%+.4f", mean(filter(>(0), gs)))))
            end
        end
    end

    # ── CSV ──
    tagc = tl_cfg * (INST_FROM_FILE ? "_I$(INST_SETFP)" : "") * pptag * (isempty(TL_INTERP_M) ? "" : "_INT") *
           (length(TL_CONDS) == 1 ? (TL_CONDS[1] == "cold_start" ? "_cold" : TL_CONDS[1] == "gnn_bias" ? "_bias" : "_gnn") : "")
    fc = joinpath(directoryBasis, "transfer_curves_$(tagc).csv")
    open(fc, "w") do io
        println(io, "instance,condition,method,rep,p,evals,best_ar,gap,ref")
        for r in TGT, p in 1:p_max
            cv = r.layers[p].curve
            for k in unique(vcat(1, collect(5:5:length(cv)), length(cv)))
                @printf(io, "%d,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f\n", r.inst, r.cond, String(r.method), r.rep, p, k,
                        -cv[k], gap_k(r, p, k), REF[(r.inst, r.cond, p)])
            end
        end
    end
    fs = joinpath(directoryBasis, "transfer_summary_$(tagc).csv")
    open(fs, "w") do io
        println(io, "p,condition,method,runs," * join(["gap_at_$(k)" for k in ks], ",") * ",zero_shot_gap,solved")
        for s in SUM
            println(io, join((s.p, s.cond, s.method, s.n, (@sprintf("%.6f", x) for x in s.gaps)...,
                              isnan(s.zero_shot) ? "" : @sprintf("%.6f", s.zero_shot), s.solved), ","))
        end
    end
    fo = joinpath(directoryBasis, "transfer_optima_$(tagc).csv")       # 画"参数集中"散点图用
    open(fo, "w") do io
        println(io, "instance,condition,p,gamma,beta,kappa_mean,kappa_max,ar")
        for g in GRAPHS, c in TL_CONDS, p in 1:p_max
            s = SRC[(g.inst, c)]; u = s.layers[p].ubest
            @printf(io, "%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", g.inst, c, p, u[1], u[2], s.kmean, s.kmax, -s.layers[p].ybest)
        end
    end
    mprintln("\n✓ 收敛曲线CSV → $(basename(fc))")
    mprintln("✓ 汇总CSV     → $(basename(fs))")
    mprintln("✓ 源最优参数CSV → $(basename(fo))   (γ·κ vs 实例, 画参数集中图)")
    fi = joinpath(directoryBasis, "transfer_instances_$(tagc).csv")    # 逐实例逐方法逐重复
    open(fi, "w") do io
        println(io, "instance,condition,method,rep,p,ar,hybrid_ar,gap_to_hybrid,greedy_chain_ar,cum_evals")
        for r in TGT, p in 1:p_max
            bv = get(BASE, (r.inst, r.cond, p), NaN)
            ce = r.method == :tl_full ? r.layers[p].nev : sum(r.layers[q].nev for q in 1:p)
            @printf(io, "%d,%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%d\n", r.inst, r.cond, String(r.method), r.rep, p,
                    -r.layers[p].ybest, bv, bv + r.layers[p].ybest, -SRC[(r.inst, r.cond)].layers[p].ybest, ce)
        end
    end
    mprintln("✓ 逐实例CSV   → $(basename(fi))")
    ff = joinpath(directoryBasis, "transfer_features_$(tagc).csv")     # 实例特征 + 源权重(画权重矩阵 / 检验特征用)
    open(ff, "w") do io
        println(io, "instance," * join(TL_FEAT_P, ",") * "," * join(TL_FEAT_G, ",") * "," *
                    join(("z_" * x for x in TL_FEAT_P), ",") * "," * join(("z_" * x for x in TL_FEAT_G), ","))
        for g in GRAPHS
            println(io, join(vcat(string(g.inst), (@sprintf("%.6f", x) for x in vcat(FEATP[g.inst], FEATG[g.inst],
                                                                                    ZP[g.inst], ZG[g.inst]))...), ","))
        end
    end
    mprintln("✓ 实例特征CSV → $(basename(ff))")
    if any(m -> m in (:tl_portp, :tl_portg), methods)
        fw = joinpath(directoryBasis, "transfer_weights_$(tagc)_H$(TL_PORT_H).csv")
        open(fw, "w") do io
            println(io, "target,source,w_phys,w_gnn")
            for g in GRAPHS
                others = [h.inst for h in GRAPHS if h.inst != g.inst]
                wp = tl_wvec(g.inst, others, :tl_portp); wg = tl_wvec(g.inst, others, :tl_portg)
                for (i, s) in enumerate(others)
                    @printf(io, "%d,%d,%.6f,%.6f\n", g.inst, s, wp[i] / maximum(wp), wg[i] / maximum(wg))
                end
            end
        end
        mprintln("✓ 源权重CSV   → $(basename(fw))   (每个目标对全部其它实例的权重, 按最大值归一)")
    end
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── omega_scan: 混合层 Rabi 幅度 Ωt 扫描(p=1 精确最优, 不受优化器影响) ─────────
# 用法:  julia r3g_sge_integrated.jl omega_scan <n_atoms> <实例数> [Ωt 列表] [网格边长]
# 例如:  ... omega_scan 12 20 1,2,4,8,15.7 60
# 每个 (Ωt, 实例, 条件): p=1 的 网格边长² 网格 + 3 个起点的 NM 精修(= 核消融/transfer_bo 的
#   p=1 标准答案算法), β 盒子按 2/Ωt 缩放(转角范围恒为 [0, 2π·盒子倍数])。
# Ωt=2 自动加入作基线, 并与 transfer_bo / 核消融缓存的 p=1 网格值核对(应一致)。
# 环境变量: QAOA_TL_CONDS(默认两种条件都跑); QAOA_INSTANCES; QAOA_BOX_SCALE。
# 注意: 本模式在每个任务里临时改写 worker 上的 OMEGA_T_REF, 与环境变量 QAOA_OMEGA_T 无关。
# ══════════════════════════════════════════════════════════════════════════
@everywhere begin
"""[omega_scan] 一个 (Ωt, 实例, 条件) 的 p=1 精确最优。"""
function om_task(tk)
    g = tk.g
    lo = [0.0, 0.0]; hi = [3π * QAOA_BOX_SCALE, π * QAOA_BOX_SCALE * 2.0 / tk.Ωt]
    OMEGA_T_REF[] = tk.Ωt                        # 本 worker 上接下来的演化都用这个 Ωt
    try
        evf = x -> begin
            xq = clamp.(x, lo, hi)
            v = try
                tk.cond == "gnn_warmstart" ?
                    loss_fn_warmstart_weighted(xq, g.best_p_gnn, g.atoms, g.edges, g.weights, g.n_atoms) :
                    loss_fn_weighted(xq, g.atoms, g.edges, g.weights, g.n_atoms)
            catch
                0.0
            end
            isfinite(v) ? v / g.cmax : 0.0
        end
        t0 = time()
        r = abl_grid_truth(evf, lo, hi, tk.n_grid)
        return (Ωt=tk.Ωt, inst=g.inst, cond=tk.cond, ra=-r.ybest, ra_grid=-r.ygrid, x=copy(r.xbest),
                edge_b=r.xbest[2] > 0.98 * hi[2], edge_g=r.xbest[1] > 0.98 * hi[1], t=time() - t0)
    finally
        OMEGA_T_REF[] = QAOA_OMEGA_T             # 恢复, 不影响同一 worker 上的其它任务
    end
end
end # @everywhere begin (omega_scan)

# ══════════════════════════════════════════════════════════════════════════
# ── 含噪声模拟 (noise_eval / noise_selftest) ──────────────────────────────
# 目的: 把无噪声下优化好的角度(迁移 checkpoint 里每个深度的 xfull)放到真实的里德堡硬件噪声下重新评估,
#       回答"深层(长脉冲)在噪声下还值不值"、"GNN 线 vs 冷启动线 vs 两线组合在噪声下谁更好"。
# 噪声模型 = QuEra 官方 BloqadeNoisy 的 Aquila 模型(lib/BloqadeNoisy/src/noise_models.jl), 参数默认值:
#   相干(每条轨迹抽一次, 整段演化不变 = 准静态):
#     Ω 全局相对涨落 0.008, Ω 逐原子相对不均匀 0.018; Δ 全局涨落 0.18 rad/μs, Δ 逐原子不均匀 0.5 rad/μs;
#     原子位置每个坐标 0.05 μm(只影响真实的 V_ij; 局部失谐 Δ_i = ½ΣV_ij 仍按**名义**位置编程 ⇒ 抵消不完美)
#   非相干(Lindblad, 用量子轨迹 MCWF 精确求解): 退相位 L = √0.02·Z_i, 弛豫 L = √0.01·|g⟩⟨r|_i (单位 1/μs)
#   读出: P(把 g 读成 r) = 0.01, P(把 r 读成 g) = 0.08 (对末态解析地计入, 无统计误差)
#   数值同 Aquila 白皮书 arXiv:2306.11727 (Ω 0.8%/2%, Δ 0.18 rad/μs, 读出 1%/8%)。
# 求解器: 自己写的无矩阵态矢量模拟器, 哈密顿量约定与 Bloqade 逐项一致:
#   H = Σ Ω_i/2 (e^{iϕ}|g⟩⟨r| + e^{−iϕ}|r⟩⟨g|) − Σ Δ_i n_i + Σ_{i<j} C6/r^6 n_i n_j
#   代价层(Ω=0)对角 ⇒ 精确指数; 混合层/态制备 ⇒ Arnoldi-Krylov(非厄米 H_eff = H − i/2 Σ L†L)带后验误差控制。
#   无噪声时与 Bloqade 的 loss_fn_* 一致到 ~1e-8(noise_selftest 会核查); MCWF 平均与精确 Lindblad 一致(同上)。
@everywhere begin
const NZ_C6 = 2π * 862690

"基组比特表: NB[b+1, i] = 基矢 b 中第 i 个原子是否在里德堡态(与 Yao/Bloqade 小端序一致)。"
nz_bits(n::Int) = Float64[(b >> (i - 1)) & 1 for b in 0:(2^n - 1), i in 1:n]

function nz_vmat(pos::Vector{Vector{Float64}})
    n = length(pos); V = zeros(n, n)
    for i in 1:n, j in i+1:n
        r2 = sum(abs2, pos[i] .- pos[j])
        V[i, j] = V[j, i] = NZ_C6 / r2^3
    end
    return V
end

"Σ_{i<j} V_ij n_i n_j 在每个基矢上的值(对角)。"
nz_eint(NB::Matrix{Float64}, V::Matrix{Float64}) = 0.5 .* vec(sum((NB * V) .* NB, dims=2))

"""脉冲序列(与 loss_fn_weighted / loss_fn_warmstart_weighted 逐段一致): 第 1 段态制备, 之后每层 代价段 + 混合段。
   返回 dur[s], Om[s, i], De[s, i], phi[s]。局部失谐的抵消项 Δt 按传入的(名义)位置计算。"""
function nz_schedule(x::Vector{Float64}, cond::String, pos::Vector{Vector{Float64}}, Ωt::Float64;
                     pg::Vector{Float64}=Float64[], eps::Float64=0.0, mixsign::Float64=1.0)
    n = length(pos); p = length(x) ÷ 2
    Ω0 = 5π; t0 = π / (2 * Ω0)
    Δt = 0.5 .* vec(sum(nz_vmat(pos), dims=2))           # = 2·sw, sw = Σ C6/(4 r^6)
    ns = 1 + 2p
    dur = zeros(ns); Om = zeros(ns, n); De = zeros(ns, n); phi = zeros(ns)
    dur[1] = t0; phi[1] = -0.5π; De[1, :] .= Δt
    if cond == "cold_start"
        Om[1, :] .= Ω0
        a = ones(n); dq = zeros(n)
    elseif cond == "gnn_warmstart"
        e = max(1e-6, eps); pc = clamp.(pg, e, 1.0 - e)
        Om[1, :] .= 2.0 .* asin.(sqrt.(pc)) ./ t0
        a = 2.0 .* sqrt.(pc .* (1.0 .- pc)); dq = mixsign .* Ωt .* (2.0 .* pc .- 1.0)
    else
        error("nz_schedule: 不支持的条件 $(cond)")
    end
    for k in 1:p
        dur[2k] = abs(x[2k-1]); De[2k, :] .= Δt                              # 代价段: Ω = 0
        dur[2k+1] = abs(x[2k]); Om[2k+1, :] .= Ωt .* a; De[2k+1, :] .= Δt .+ dq   # 混合段
    end
    return (dur=dur, Om=Om, De=De, phi=phi)
end

"y = H ψ, H = diag(d) + Σ_i Ω_i/2 (e^{iϕ}|g⟩⟨r| + e^{−iϕ}|r⟩⟨g|)_i (d 可以是复数 ⇒ 非厄米)。"
function nz_matvec!(y::AbstractVector{ComplexF64}, ψ::AbstractVector{ComplexF64}, d::Vector{ComplexF64},
                    Om::Vector{Float64}, ϕ::Float64, n::Int)
    dim = length(ψ)
    @inbounds for b in 1:dim
        y[b] = d[b] * ψ[b]
    end
    cm = cis(-ϕ); cp = cis(ϕ)
    @inbounds for i in 1:n
        Om[i] == 0.0 && continue
        m = 1 << (i - 1); hm = 0.5 * Om[i] * cm; hp = 0.5 * Om[i] * cp
        for b in 0:dim-1
            (b & m) != 0 && continue
            b1 = b | m
            y[b1+1] += hm * ψ[b+1]          # |g⟩ → |r⟩: e^{−iϕ}
            y[b+1]  += hp * ψ[b1+1]         # |r⟩ → |g⟩: e^{+iϕ}
        end
    end
    return y
end

"""w ← exp(−i H τ) w (H 非厄米也可以)。Arnoldi(CGS2 正交化), 子步长自适应:
   误差估计 = β·|[exp(dt·Ĥ)]_{k+1,1}|(增广 Hessenberg 的修正项), 超过 tol·β 就把子步减半。
   对角部分先减去实部中点(只差一个整体相位), 降低 ‖H‖。"""
function nz_expv!(w::Vector{ComplexF64}, τ::Float64, d::Vector{ComplexF64}, Om::Vector{Float64},
                  ϕ::Float64, n::Int; m::Int=20, tol::Float64=1e-10)   # m=20: 单位时间的正交化开销 ∝ m, 比 30 省
    τ <= 0.0 && return w
    dr = real.(d); shift = 0.5 * (maximum(dr) + minimum(dr))
    dd = d .- shift
    normA = maximum(abs, dd) + 0.5 * sum(abs, Om)
    dim = length(w)
    V = Matrix{ComplexF64}(undef, dim, m + 1); H = zeros(ComplexF64, m + 1, m)
    v = Vector{ComplexF64}(undef, dim)
    t = 0.0; dt_try = min(τ, m / (1.5 * max(normA, 1e-12)))
    while t < τ
        rem = τ - t
        β = norm(w); β == 0.0 && return w
        fill!(H, 0.0)
        @views V[:, 1] .= w ./ β
        k = m; happy = false
        for j in 1:m
            @views nz_matvec!(v, V[:, j], dd, Om, ϕ, n)
            v .*= -im
            Vj = @view V[:, 1:j]
            h1 = Vj' * v; mul!(v, Vj, h1, -one(ComplexF64), one(ComplexF64))
            h2 = Vj' * v; mul!(v, Vj, h2, -one(ComplexF64), one(ComplexF64))
            @views H[1:j, j] .= h1 .+ h2
            hn = norm(v); H[j+1, j] = hn
            if hn < 1e-12 * max(normA, 1.0)
                k = j; happy = true; break
            end
            @views V[:, j+1] .= v ./ hn
        end
        dt = happy ? rem : min(rem, dt_try)
        u = similar(w)
        while true
            if happy
                F = exp(dt .* H[1:k, 1:k]); @views mul!(u, V[:, 1:k], F[:, 1]); u .*= β; err = 0.0
            else
                Ha = zeros(ComplexF64, k + 1, k + 1); Ha[1:k, 1:k] .= H[1:k, 1:k]; Ha[k+1, k] = H[k+1, k]
                F = exp(dt .* Ha); @views mul!(u, V[:, 1:k+1], F[:, 1]); u .*= β; err = β * abs(F[k+1, 1])
            end
            (err <= tol * β || dt < 1e-13) && break
            dt *= 0.5
        end
        w .= u .* cis(-shift * dt)
        if dt >= rem
            t = τ
        else
            t += dt
            dt_try = min(2dt, m / (0.5 * max(normA, 1e-12)))
        end
    end
    return w
end

"按非负权重 w 抽一个下标。"
function nz_pick(rng, w::AbstractVector{Float64})
    s = sum(w); u = rand(rng) * s; c = 0.0
    for i in eachindex(w)
        c += w[i]; u <= c && return i
    end
    return lastindex(w)
end

"""一条量子轨迹(MCWF, 等待时间算法)。sch = 已含相干噪声的脉冲序列; E = 真实(含位置噪声)的相互作用对角项。
   弛豫 |r⟩→|g⟩: H_eff = H − i·grel/2·Σ n_i, 范数掉到随机阈值 r 时跳(二分定位跳跃时刻, 按 ⟨n_q⟩ 选原子)。
   退相位 √gdep·Z_i: L†L ∝ 恒等 ⇒ 跳跃与态无关, 等价于总速率 n·gdep 的泊松过程, 预先抽好时刻和原子。
   grel = gdep = 0 ⇒ 纯幺正演化(无噪声/只有相干噪声时用)。返回(归一化末态, 弛豫跳跃次数)。"""
function nz_traj(sch, E::Vector{Float64}, NB::Matrix{Float64}, Nb::Vector{Float64}, n::Int,
                 grel::Float64, gdep::Float64, rng)
    dim = size(NB, 1)
    ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
    T = sum(sch.dur)
    zt = Float64[]; zq = Int[]
    if gdep > 0
        tt = randexp(rng) / (n * gdep)
        while tt < T
            push!(zt, tt); push!(zq, rand(rng, 1:n)); tt += randexp(rng) / (n * gdep)
        end
    end
    r = rand(rng); clock = 0.0; nj = 0; zi = 1
    for s in eachindex(sch.dur)
        τ = sch.dur[s]; τ <= 0.0 && continue
        Om = sch.Om[s, :]; De = sch.De[s, :]; ϕ = sch.phi[s]
        d = ComplexF64.(E .- NB * De) .- (0.5im * grel) .* Nb
        isdiag = all(iszero, Om)
        prop = (ps, tl) -> isdiag ? ps .* exp.((-im * tl) .* d) : nz_expv!(copy(ps), tl, d, Om, ϕ, n)
        tend = clock + τ
        while clock < tend - 1e-12
            tz = zi <= length(zt) ? zt[zi] : Inf
            tstop = min(tz, tend); dτ = max(tstop - clock, 0.0)
            ps1 = prop(ψ, dτ)
            if grel > 0 && real(dot(ps1, ps1)) < r
                lo = 0.0; hi = dτ
                for _ in 1:40
                    mid = 0.5 * (lo + hi); pm = prop(ψ, mid)
                    if real(dot(pm, pm)) < r
                        hi = mid
                    else
                        lo = mid
                    end
                    hi - lo < 1e-8 && break
                end
                pj = prop(ψ, hi)
                q = nz_pick(rng, NB' * abs2.(pj))
                mq = 1 << (q - 1); new = zeros(ComplexF64, dim)
                @inbounds for b in 0:dim-1
                    (b & mq) != 0 && (new[(b ⊻ mq)+1] = pj[b+1])
                end
                ψ = new ./ norm(new); r = rand(rng); clock += hi; nj += 1
                continue
            end
            ψ = ps1; clock = tstop
            if tstop == tz                                     # 退相位跳跃: Z_q
                mq = 1 << (zq[zi] - 1)
                @inbounds for b in 0:dim-1
                    (b & mq) != 0 && (ψ[b+1] = -ψ[b+1])
                end
                zi += 1
            end
        end
    end
    return ψ ./ norm(ψ), nj
end

"""末态的期望割: (理想测量, 计入读出误差)。读出: 每个原子独立地 P(0→1)=p01, P(1→0)=p10;
   测得的 n' 满足 E[n_i'] = p01 + a⟨n_i⟩, E[n_i'n_j'] = p01² + p01·a(⟨n_i⟩+⟨n_j⟩) + a²⟨n_i n_j⟩, a = 1−p01−p10。"""
function nz_cut(ψ::Vector{ComplexF64}, NB::Matrix{Float64}, edges, w::Vector{Float64}, p01::Float64, p10::Float64)
    pr = abs2.(ψ); pr ./= sum(pr)
    nm = NB' * pr
    a = 1.0 - p01 - p10; ideal = 0.0; meas = 0.0
    for (k, (u, v)) in enumerate(edges)
        nn = 0.0
        @inbounds for b in eachindex(pr)
            nn += pr[b] * NB[b, u] * NB[b, v]
        end
        ideal += w[k] * (nm[u] + nm[v] - 2nn)
        mu = p01 + a * nm[u]; mv = p01 + a * nm[v]
        mm = p01^2 + p01 * a * (nm[u] + nm[v]) + a^2 * nn
        meas += w[k] * (mu + mv - 2mm)
    end
    return ideal, meas
end

"读出误差作用到整个测量分布上(逐原子 2×2 混淆矩阵, 原地): P(测得 1|真 0)=p01, P(测得 0|真 1)=p10。"
function nz_confuse!(q::Vector{Float64}, p01::Float64, p10::Float64, n::Int)
    dim = length(q)
    @inbounds for i in 1:n
        m = 1 << (i - 1)
        for b in 0:dim-1
            (b & m) != 0 && continue
            a0 = q[b+1]; a1 = q[(b|m)+1]
            q[b+1] = (1.0 - p01) * a0 + p10 * a1
            q[(b|m)+1] = p01 * a0 + (1.0 - p10) * a1
        end
    end
    return q
end

"每个基矢(比特串)的割值。"
function nz_cutvec(NB::Matrix{Float64}, edges, w::Vector{Float64})
    c = zeros(size(NB, 1))
    for (k, (u, v)) in enumerate(edges)
        c .+= w[k] .* (NB[:, u] .+ NB[:, v] .- 2.0 .* NB[:, u] .* NB[:, v])
    end
    return c
end

const NZ_SHOTS = (10, 100, 1000)
"""采样指标: 测量分布 Q 下 (采到最优割的概率, N 次采样中最好割的期望/C_max, N ∈ NZ_SHOTS)。
   E[max of N] = Σ_k v_k (F(v_k)^N − F(v_{k−1})^N), F = 割值的累积分布。"""
function nz_sampling(Q::Vector{Float64}, cutv::Vector{Float64}, cmax::Float64)
    popt = sum(Q[b] for b in eachindex(Q) if cutv[b] >= cmax - 1e-9; init=0.0)
    perm = sortperm(cutv); cs = cutv[perm]; qs = Q[perm]
    vals = Float64[]; F = Float64[]; acc = 0.0
    for k in eachindex(cs)
        acc += qs[k]
        if k == length(cs) || cs[k+1] > cs[k] + 1e-9
            push!(vals, cs[k]); push!(F, min(acc, 1.0))
        end
    end
    best = Float64[]
    for N in NZ_SHOTS
        e = 0.0; Fp = 0.0
        for k in eachindex(vals)
            e += vals[k] * (F[k]^N - Fp^N); Fp = F[k]
        end
        push!(best, e / cmax)
    end
    return popt, best
end

"""抽一份准静态相干噪声: 返回(带噪脉冲序列, 带噪原子位置)。抽样顺序固定 ⇒ 同一 rng 种子在不同 p/条件下
   得到同一份噪声(公共随机数, 配对比较的方差更小)。"""
function nz_coherent(rng, sch, pos::Vector{Vector{Float64}}, cfg)
    n = length(pos)
    fΩ = 1.0 .+ cfg.dOg * randn(rng) .+ cfg.dOl .* randn(rng, n)
    dΔ = cfg.dDg * randn(rng) .+ cfg.dDl .* randn(rng, n)
    posn = [p .+ cfg.dx .* randn(rng, length(p)) for p in pos]
    return (dur=sch.dur, Om=sch.Om .* fΩ', De=sch.De .+ dΔ', phi=sch.phi), posn
end

"""一个评估任务: 某实例、某条件、某深度的角度 x, 在噪声模型 cfg 下跑 ntraj 条轨迹。
   cfg.coh / cfg.inc 控制是否开相干 / 非相干噪声; 两者都关时只算 1 次(只剩读出误差, 解析)。"""
function nz_task(tk)
    t0 = time()
    pos = tk.pos; n = length(pos); NB = nz_bits(n); Nb = vec(sum(NB, dims=2)); cfg = tk.cfg
    sch = nz_schedule(tk.x, tk.cond, pos, tk.Ωt; pg=tk.pg, eps=tk.eps, mixsign=tk.mixsign)
    E0 = nz_eint(NB, nz_vmat(pos))
    ψ0, _ = nz_traj(sch, E0, NB, Nb, n, 0.0, 0.0, MersenneTwister(0))
    c0, c0ro = nz_cut(ψ0, NB, tk.edges, tk.weights, cfg.p01, cfg.p10)
    stoch = cfg.coh || cfg.inc
    nt = stoch ? tk.ntraj : 1
    ra = zeros(nt); raro = zeros(nt); njs = zeros(nt)
    dim = size(NB, 1); Qsum = zeros(dim)                      # 含噪测量分布(对轨迹平均 = 混态的测量分布)
    pr0 = abs2.(ψ0); pr0 ./= sum(pr0)
    for k in 1:nt
        if !stoch
            ra[k] = c0; raro[k] = c0ro; Qsum .+= pr0; continue
        end
        rc = MersenneTwister(hash((tk.inst, k, tk.seed, 1)))                     # 与 p、条件无关 ⇒ 公共随机数(只哈希整数/字符串: Symbol 的哈希跨进程不稳定)
        rj = MersenneTwister(hash((tk.inst, k, tk.seed, 2, tk.cond, tk.p)))
        schn, posn = cfg.coh ? nz_coherent(rc, sch, pos, cfg) : (sch, pos)
        E = cfg.coh ? nz_eint(NB, nz_vmat(posn)) : E0
        ψ, nj = nz_traj(schn, E, NB, Nb, n, cfg.inc ? cfg.grel : 0.0, cfg.inc ? cfg.gdep : 0.0, rj)
        ra[k], raro[k] = nz_cut(ψ, NB, tk.edges, tk.weights, cfg.p01, cfg.p10)
        njs[k] = nj
        Qsum .+= abs2.(ψ)
    end
    Q = Qsum ./ sum(Qsum); nz_confuse!(Q, cfg.p01, cfg.p10, n)      # 计入读出误差 = 硬件实际的比特串分布
    cutv = nz_cutvec(NB, tk.edges, tk.weights)
    popt, best = nz_sampling(Q, cutv, tk.cmax)                      # 含噪 + 读出
    popt0, best0 = nz_sampling(pr0, cutv, tk.cmax)                  # 无噪声理想参照
    poptu, bestu = nz_sampling(fill(1.0 / dim, dim), cutv, tk.cmax) # 均匀随机猜的基线
    return (label=tk.label, cond=tk.cond, inst=tk.inst, rep=tk.rep, p=tk.p, model=tk.model, scale=tk.scale,
            ntraj=nt, T=sum(sch.dur), ra0=c0 / tk.cmax, ra0ro=c0ro / tk.cmax, ra_stored=tk.ra_stored,
            ra=ra ./ tk.cmax, raro=raro ./ tk.cmax, nj=mean(njs),
            popt=popt, best=best, popt0=popt0, best0=best0, poptu=poptu, bestu=bestu, t=time() - t0)
end

# ── 自检用: 稠密哈密顿量与精确 Lindblad(只适合 n ≤ 5) ──
function nz_hdense(Om::Vector{Float64}, De::Vector{Float64}, ϕ::Float64, E::Vector{Float64}, NB::Matrix{Float64}, n::Int)
    dim = 2^n; H = Matrix{ComplexF64}(Diagonal(ComplexF64.(E .- NB * De)))
    for i in 1:n
        m = 1 << (i - 1)
        for b in 0:dim-1
            (b & m) != 0 && continue
            H[(b|m)+1, b+1] += 0.5 * Om[i] * cis(-ϕ)
            H[b+1, (b|m)+1] += 0.5 * Om[i] * cis(ϕ)
        end
    end
    return H
end

function nz_lindblad_cut(sch, pos, n, grel, gdep, edges, w)
    dim = 2^n; NB = nz_bits(n); E = nz_eint(NB, nz_vmat(pos)); Id = Matrix{ComplexF64}(LinearAlgebra.I, dim, dim)
    Ls = Matrix{ComplexF64}[]
    for i in 1:n
        m = 1 << (i - 1)
        push!(Ls, sqrt(gdep) .* Matrix{ComplexF64}(Diagonal(ComplexF64.(1.0 .- 2.0 .* NB[:, i]))))
        S = zeros(ComplexF64, dim, dim)
        for b in 0:dim-1
            (b & m) != 0 && (S[(b ⊻ m)+1, b+1] = 1.0)
        end
        push!(Ls, sqrt(grel) .* S)
    end
    ρ = zeros(ComplexF64, dim, dim); ρ[1, 1] = 1.0; vρ = vec(ρ)
    for s in eachindex(sch.dur)
        sch.dur[s] <= 0 && continue
        H = nz_hdense(sch.Om[s, :], sch.De[s, :], sch.phi[s], E, NB, n)
        Lsup = -im .* (kron(Id, H) .- kron(transpose(H), Id))
        for L in Ls
            LdL = L' * L
            Lsup .+= kron(conj.(L), L) .- 0.5 .* (kron(Id, LdL) .+ kron(transpose(LdL), Id))
        end
        vρ = exp(sch.dur[s] .* Lsup) * vρ
    end
    pr = real.(diag(reshape(vρ, dim, dim)))
    return sum(w[k] * sum(pr .* (NB[:, u] .+ NB[:, v] .- 2.0 .* NB[:, u] .* NB[:, v])) for (k, (u, v)) in enumerate(edges))
end
end # @everywhere begin (含噪声模拟)

# 噪声参数: Aquila 默认值(BloqadeNoisy), 可用 QAOA_NOISE_PARAMS="dOg=0.008,dDl=0.5,..." 覆盖
const NZ_BASE_DEFAULT = Dict("dOg" => 0.008, "dOl" => 0.018, "dDg" => 0.18, "dDl" => 0.5, "dx" => 0.05,
                             "gdep" => 0.02, "grel" => 0.01, "p01" => 0.01, "p10" => 0.08)
function nz_base_params()
    b = copy(NZ_BASE_DEFAULT)
    for kv in split(get(ENV, "QAOA_NOISE_PARAMS", ""), ",")
        s = strip(kv); isempty(s) && continue
        k, v = strip.(split(s, "="))
        haskey(b, k) || error("QAOA_NOISE_PARAMS 里的未知参数 $(k); 可选: $(join(sort(collect(keys(b))), ","))")
        b[k] = parse(Float64, v)
    end
    return b
end
"模型 full/coh/inc × 强度倍数 λ: 所有涨落幅度、速率、读出错误率都乘 λ。"
function nz_cfg(b::Dict, model::String, λ::Float64)
    model in ("full", "coh", "inc") || error("噪声模型只能是 full / coh / inc, 不能是 $(model)")
    coh = model != "inc"; inc = model != "coh"
    return (coh=coh, inc=inc, dOg=λ * b["dOg"], dOl=λ * b["dOl"], dDg=λ * b["dDg"], dDl=λ * b["dDl"], dx=λ * b["dx"],
            gdep=λ * b["gdep"], grel=λ * b["grel"], p01=λ * b["p01"], p10=λ * b["p10"])
end
"源目录名里的设置: _ws<ε> → ε, _wsE → Egger 精确 mixer(−1), _om<Ωt> → Ωt(没有则 2)。"
function nz_dir_settings(dir::String)
    m1 = match(r"_ws(\d+(?:\.\d+)?)", dir); m2 = match(r"_om(\d+(?:\.\d+)?)", dir)
    return (eps = m1 === nothing ? 0.0 : parse(Float64, m1.captures[1]),
            mixsign = occursin("_wsE", dir) ? -1.0 : 1.0,
            Ωt = m2 === nothing ? 2.0 : parse(Float64, m2.captures[1]))
end
nz_pos(g) = [Float64[c for c in g.atoms[i]] for i in 1:g.n_atoms]

function nz_selftest_main()
    n_at        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    n_instances = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
    optimizer = "zygote"
    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" * (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    isfile(prior_file) || error("找不到先验 $(prior_tag)")
    gnn_params = deserialize(prior_file)
    GRAPHS, _, _, _ = obtain_test_instances(n_at, n_instances, gnn_params; verbose=false, save=false, prior_tag=prior_tag)
    Ωt = OMEGA_T_REF[]
    println("\n" * "="^70)
    println("噪声模拟器自检  (Ωt = $(Ωt), GNN 预热: ε = $(WS_EPS_G), mixer 符号 = $(WS_MIXSIGN_G))")
    println("="^70)
    # (1) 无噪声: 自写模拟器 vs Bloqade 的 loss_fn_*(同一组随机角度)
    rng = MersenneTwister(2024); worst = 0.0
    for g in GRAPHS[1:min(3, end)], p in (1, 3), c in ("cold_start", "gnn_warmstart")
        x = Float64[isodd(j) ? 2.0 * rand(rng) : (2π / Ωt) * rand(rng) for j in 1:2p]
        pos = nz_pos(g); NB = nz_bits(g.n_atoms); Nb = vec(sum(NB, dims=2))
        sch = nz_schedule(x, c, pos, Ωt; pg=Vector{Float64}(g.best_p_gnn), eps=WS_EPS_G, mixsign=WS_MIXSIGN_G)
        ψ, _ = nz_traj(sch, nz_eint(NB, nz_vmat(pos)), NB, Nb, g.n_atoms, 0.0, 0.0, MersenneTwister(0))
        mine = nz_cut(ψ, NB, g.edges, g.weights, 0.0, 0.0)[1]
        ref = c == "cold_start" ? -loss_fn_weighted(x, g.atoms, g.edges, g.weights, g.n_atoms) :
              -loss_fn_warmstart_weighted(x, g.best_p_gnn, g.atoms, g.edges, g.weights, g.n_atoms; mixsign=WS_MIXSIGN_G)
        worst = max(worst, abs(mine - ref) / g.cmax)
        @printf("  inst%-2d p=%d %-13s  自写 %.8f   Bloqade %.8f   差 %.1e\n", g.inst, p, c, mine / g.cmax, ref / g.cmax, abs(mine - ref) / g.cmax)
    end
    println(worst < 1e-5 ? @sprintf("(1) 无噪声一致性: 最大差 %.1e ✓", worst) :
            @sprintf("(1) ⚠ 无噪声一致性: 最大差 %.1e —— 哈密顿量约定不一致, 先别跑 noise_eval", worst))
    # (2) Arnoldi(非厄米)vs 稠密矩阵指数: 取实例 1 的前 8 个原子
    g = GRAPHS[1]; n8 = min(8, g.n_atoms); pos8 = nz_pos(g)[1:n8]; NB8 = nz_bits(n8)
    E8 = nz_eint(NB8, nz_vmat(pos8)); De8 = 0.5 .* vec(sum(nz_vmat(pos8), dims=2)) .+ 3.0 .* randn(rng, n8)
    Om8 = Ωt .* (0.5 .+ 0.5 .* rand(rng, n8)); d8 = ComplexF64.(E8 .- NB8 * De8) .- 0.15im .* vec(sum(NB8, dims=2))
    v0 = randn(rng, ComplexF64, 2^n8); v0 ./= norm(v0)
    Hd = Matrix{ComplexF64}(Diagonal(d8)) .+ (nz_hdense(Om8, zeros(n8), 0.3, zeros(2^n8), NB8, n8))
    e2 = maximum(abs, nz_expv!(copy(v0), 0.4, d8, Om8, 0.3, n8) .- exp(-im * 0.4 .* Hd) * v0)
    println(e2 < 1e-8 ? @sprintf("(2) Arnoldi vs 稠密 exp (n=%d, 非厄米): 最大差 %.1e ✓", n8, e2) :
            @sprintf("(2) ⚠ Arnoldi vs 稠密 exp: 最大差 %.1e", e2))
    # (3) MCWF vs 精确 Lindblad: 实例 1 的前 4 个原子, 夸大的速率, 两种条件
    n4 = 4; pos4 = nz_pos(g)[1:n4]
    e4 = [(u, v) for (u, v) in g.edges if u <= n4 && v <= n4]; w4 = [g.weights[k] for (k, (u, v)) in enumerate(g.edges) if u <= n4 && v <= n4]
    if isempty(e4)
        e4 = [(1, 2), (2, 3), (3, 4)]; w4 = [1.0, 1.0, 1.0]
    end
    NB4 = nz_bits(n4); Nb4 = vec(sum(NB4, dims=2)); E4 = nz_eint(NB4, nz_vmat(pos4))
    x4 = [0.6, 0.15, 0.9, 0.3]; grel4 = 0.4; gdep4 = 0.3; ntr = 3000
    for c in ("cold_start", "gnn_warmstart")
        sch4 = nz_schedule(x4, c, pos4, Ωt; pg=Vector{Float64}(g.best_p_gnn[1:n4]), eps=max(WS_EPS_G, 0.2), mixsign=WS_MIXSIGN_G)
        ex = nz_lindblad_cut(sch4, pos4, n4, grel4, gdep4, e4, w4)
        rj = MersenneTwister(7)
        vals = [nz_cut(nz_traj(sch4, E4, NB4, Nb4, n4, grel4, gdep4, rj)[1], NB4, e4, w4, 0.0, 0.0)[1] for _ in 1:ntr]
        z = (mean(vals) - ex) / (std(vals) / sqrt(ntr))
        @printf("(3) MCWF vs Lindblad (n=4, %s, γ_rel=%.1f, γ_deph=%.1f): 精确 %.5f  轨迹平均 %.5f ± %.5f  (z = %+.2f) %s\n",
                c, grel4, gdep4, ex, mean(vals), std(vals) / sqrt(ntr), z, abs(z) < 3.5 ? "✓" : "⚠")
    end
    # (4) 计时: 实例 1, 冷启动 p=10 随机角度, 全部噪声(Aquila 强度)
    cfg = nz_cfg(nz_base_params(), "full", 1.0); pos = nz_pos(g); NB = nz_bits(g.n_atoms); Nb = vec(sum(NB, dims=2))
    x10 = Float64[isodd(j) ? 0.8 * rand(rng) : (2π / Ωt) * rand(rng) for j in 1:20]
    sch = nz_schedule(x10, "cold_start", pos, Ωt)
    tt = @elapsed for k in 1:5
        schn, posn = nz_coherent(MersenneTwister(k), sch, pos, cfg)
        nz_traj(schn, nz_eint(NB, nz_vmat(posn)), NB, Nb, g.n_atoms, cfg.grel, cfg.gdep, MersenneTwister(100 + k))
    end
    @printf("(4) 计时: 冷启动 p=10(总时长 %.1f μs)每条轨迹 %.2f 秒(含编译, 正式运行会更快)\n", sum(sch.dur), tt / 5)
    println("="^70)
end
if length(ARGS) >= 1 && ARGS[1] == "noise_selftest"
    nz_selftest_main()
    exit(0)
end

function nz_eval_main()
    n_at        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    n_instances = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
    NZ_NTRAJ    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 100
    NZ_SEED     = parse(Int, get(ENV, "QAOA_NOISE_SEED", "1"))
    NZ_PLIST    = parse.(Int, split(get(ENV, "QAOA_NOISE_P", "1,2,3,5,10,15,20"), ","))
    NZ_REPS     = parse.(Int, split(get(ENV, "QAOA_NOISE_REP", "1"), ","))
    NZ_MODELS   = [(String(split(s, ":")[1]), parse(Float64, split(s, ":")[2]))
                   for s in split(get(ENV, "QAOA_NOISE_MODELS", "full:1,coh:1,inc:1,full:0.5,full:2,full:0.1,full:0.2,full:0.3"), ",")]
    ("full", 1.0) in NZ_MODELS || pushfirst!(NZ_MODELS, ("full", 1.0))
    NZ_BASE = nz_base_params()
    sfx_src = "_om$(QAOA_OMEGA_T)"
    NZ_SRC_DEFAULT = "cold=ckpt_tl_n$(n_at)_S80_P20_T30_K19_mean$(sfx_src)|cold_start|tl_annx_S30x2P150_M2x30p1_GA1.0_I;" *
                     "gnn=ckpt_tl_n$(n_at)_S80_P10_T30_K19_mean_ws0.2_wsE$(sfx_src)|gnn_warmstart|tl_hyb_S30x2P150_M2x30p1"
    NZ_SRC = NamedTuple[]
    for item in split(get(ENV, "QAOA_NOISE_SRC", NZ_SRC_DEFAULT), ";")
        s = strip(item); isempty(s) && continue
        lab, rest = split(s, "="; limit=2); parts = split(rest, "|")
        length(parts) == 3 || error("QAOA_NOISE_SRC 每一项写成 标签=目录|条件|方法标签, 收到: $(s)")
        push!(NZ_SRC, (label=String(strip(lab)), dir=String(strip(parts[1])), cond=String(strip(parts[2])), tag=String(strip(parts[3]))))
    end
    optimizer = "zygote"
    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" * (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    isfile(prior_file) || error("找不到先验 $(prior_tag)")
    gnn_params = deserialize(prior_file)
    GRAPHS, INST_FPS, INST_SETFP, INST_FROM_FILE =
        obtain_test_instances(n_at, n_instances, gnn_params; verbose=false, save=false, prior_tag=prior_tag)
    GBY = Dict(g.inst => g for g in GRAPHS)

    println("\n" * "="^70)
    println("含噪声评估: 无噪声下优化好的角度, 放到里德堡硬件噪声下重新评估")
    @printf("噪声模型(λ=1 即 Aquila, BloqadeNoisy 官方参数): Ω 全局 %.3f / 逐原子 %.3f (相对); Δ 全局 %.2f / 逐原子 %.2f rad/μs; 位置 %.2f μm\n",
            NZ_BASE["dOg"], NZ_BASE["dOl"], NZ_BASE["dDg"], NZ_BASE["dDl"], NZ_BASE["dx"])
    @printf("                 退相位 %.3f /μs (√γ·Z), 弛豫 %.3f /μs (|g⟩⟨r|); 读出 P(g→r)=%.2f, P(r→g)=%.2f\n",
            NZ_BASE["gdep"], NZ_BASE["grel"], NZ_BASE["p01"], NZ_BASE["p10"])
    println("模型: " * join(["$(m)×$(λ)" for (m, λ) in NZ_MODELS], ", ") * "   (full = 相干+非相干, coh = 只有准静态相干, inc = 只有 Lindblad; 读出误差另列)")
    println("每个(实例, 深度, 模型) $(NZ_NTRAJ) 条轨迹; 相干噪声用公共随机数(同一实例的所有深度/两条线共用同一组噪声样本)")
    println("深度: $(join(NZ_PLIST, ","))   重复: $(join(NZ_REPS, ","))")
    println("="^70)

    # ── 读入角度 ──
    TASKS = NamedTuple[]; NZ_FIX = String[]
    nz_ideal_ra(x, cond, g, st) = (pos = nz_pos(g); NB = nz_bits(g.n_atoms); Nb = vec(sum(NB, dims=2));
        sch = nz_schedule(x, cond, pos, st.Ωt; pg=Vector{Float64}(g.best_p_gnn), eps=st.eps, mixsign=st.mixsign);
        nz_cut(nz_traj(sch, nz_eint(NB, nz_vmat(pos)), NB, Nb, g.n_atoms, 0.0, 0.0, MersenneTwister(0))[1], NB, g.edges, g.weights, 0.0, 0.0)[1] / g.cmax)
    for src in NZ_SRC
        dpath = joinpath(directoryBasis, src.dir)
        isdir(dpath) || error("找不到源目录 $(src.dir)  (QAOA_NOISE_SRC 的 $(src.label))。输出目录里现有的迁移目录: " *
                              join(filter(d -> startswith(d, "ckpt_tl_n"), readdir(directoryBasis)), "  "))
        st = nz_dir_settings(src.dir)
        abs(st.Ωt - QAOA_OMEGA_T) < 1e-9 || println("  ⚠ $(src.label): 目录的 Ωt=$(st.Ωt) 与本次 QAOA_OMEGA_T=$(QAOA_OMEGA_T) 不同(评估按目录的 Ωt)")
        rx = Regex("^t_i(\\d+)_" * src.cond * "_" * replace(src.tag, "." => "\\.") * "_r(\\d+)\\.jls\$")
        files = filter(f -> occursin(rx, f), readdir(dpath))
        if isempty(files)
            cand = unique([replace(f, r"^t_i\d+_" => "", r"_r\d+\.jls$" => "") for f in readdir(dpath) if startswith(f, "t_i") && endswith(f, ".jls")])
            error("$(src.label): 目录 $(src.dir) 里没有 条件=$(src.cond) 方法标签=$(src.tag) 的文件。目录里有的(条件_方法标签): " *
                  join(cand[1:min(end, 12)], "  "))
        end
        nfound = 0; pmax_src = 0
        for f in files
            m = match(rx, f); inst = parse(Int, m.captures[1]); rep = parse(Int, m.captures[2])
            (haskey(GBY, inst) && rep in NZ_REPS) || continue
            r = try deserialize(joinpath(dpath, f)) catch; nothing end
            r === nothing && continue
            g = GBY[inst]; pmax_src = max(pmax_src, length(r.layers)); nfound += 1
            hiL(q) = repeat([3π * QAOA_BOX_SCALE, π * QAOA_BOX_SCALE * 2.0 / st.Ωt], q)
            for L in r.layers
                L.p in NZ_PLIST || continue
                xL = clamp.(Vector{Float64}(L.xfull), 0.0, hiL(L.p))
                # 自写模拟器复核这组角度的无噪声值; 对不上(已知: Bloqade 对极长代价段 + 混合段个别情况算错, 优化器"看到"的
                # 值其实属于更浅层的角度)就改用存的值相同的更浅层角度并补零层 —— 补零层精确等价, 存的 R_a 由它真实达到
                if abs(nz_ideal_ra(xL, src.cond, g, st) + L.ybest) > 1e-5
                    sub = nothing
                    for q in L.p-1:-1:1                       # 同一条向量: 从第 q+1 层起全部置零(零层 = 恒等), 找最深的一致者
                        xq = vcat(xL[1:2q], zeros(2 * (L.p - q)))
                        if abs(nz_ideal_ra(xq, src.cond, g, st) + L.ybest) < 1e-5
                            sub = (q, xq); break
                        end
                    end
                    if sub === nothing
                        push!(NZ_FIX, @sprintf("    %s inst%d r%d p=%d: 存的角度复算对不上, 也没找到等价的浅层角度 —— 照原角度评估(核查会报 ⚠)", src.label, inst, rep, L.p))
                    else
                        xL = sub[2]
                        push!(NZ_FIX, @sprintf("    %s inst%d r%d p=%d: 存的角度复算对不上(Bloqade 个别情况算错), 保留前 %d 层、其后置零(存的 R_a=%.4f 由它精确达到)",
                                               src.label, inst, rep, L.p, sub[1], -L.ybest))
                    end
                end
                for (mdl, λ) in NZ_MODELS
                    push!(TASKS, (label=src.label, cond=src.cond, inst=inst, rep=rep, p=L.p, model=mdl, scale=λ,
                                  x=xL, ra_stored=-L.ybest, pos=nz_pos(g), edges=g.edges,
                                  weights=g.weights, cmax=g.cmax, pg=Vector{Float64}(g.best_p_gnn), eps=st.eps,
                                  mixsign=st.mixsign, Ωt=st.Ωt, ntraj=NZ_NTRAJ, seed=NZ_SEED,
                                  cfg=nz_cfg(NZ_BASE, mdl, λ)))
                end
            end
        end
        @printf("  源 %-6s: %s  条件=%s  方法=%s  (%d 个文件, 深度到 %d; ε=%g, mixer 符号=%+d, Ωt=%g)\n",
                src.label, src.dir, src.cond, src.tag, nfound, pmax_src, st.eps, Int(st.mixsign), st.Ωt)
    end
    isempty(TASKS) && error("没有可评估的任务(检查 QAOA_NOISE_SRC / QAOA_NOISE_P / QAOA_NOISE_REP)")
    if !isempty(NZ_FIX)
        println("  角度替换(无噪声复算与 checkpoint 不符的层):")
        foreach(println, NZ_FIX)
    end

    # ── checkpoint ──
    ptag = NZ_BASE == NZ_BASE_DEFAULT ? "" : "_P" * string(hash(sort(collect(NZ_BASE))) % 0xffffff, base=16)
    nz_dir = joinpath(directoryBasis, "ckpt_noise_n$(n_at)_T$(NZ_NTRAJ)_s$(NZ_SEED)$(ptag)" * (QAOA_OMEGA_T == 2.0 ? "" : "_om$(QAOA_OMEGA_T)"))
    mkpath(nz_dir)
    check_ckpt_fingerprints(nz_dir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE)
    nz_fname(t) = "nz_$(t.label)_i$(t.inst)_r$(t.rep)_p$(t.p)_$(t.model)$(t.scale).jls"
    RES = Dict{Tuple,Any}(); TODO = NamedTuple[]
    for t in TASKS
        f = joinpath(nz_dir, nz_fname(t))
        r = isfile(f) ? (try deserialize(f) catch; nothing end) : nothing
        # 缓存里无噪声值与 checkpoint 对不上(旧版没把角度夹回参数盒)的任务重算
        # full 模型的旧缓存没有采样指标(popt 等) ⇒ 重算
        (r === nothing || abs(r.ra0 - r.ra_stored) > 1e-5 || (t.model == "full" && !hasproperty(r, :popt))) ? push!(TODO, t) :
            (RES[(t.label, t.inst, t.rep, t.p, t.model, t.scale)] = r)
    end
    sort!(TODO, by = t -> -(t.p * (t.model == "coh" ? 0.5 : 1.0)))          # 长任务先发, 负载更均衡
    @printf("任务: 共 %d 个, 已缓存 %d, 本次计算 %d (worker=%d)\n", length(TASKS), length(RES), length(TODO), nworkers())
    t_nz = time(); odir = nz_dir
    RR = isempty(TODO) ? Any[] : pmap(TODO) do tk
        r = nz_task(tk)
        try serialize(joinpath(odir, "nz_$(tk.label)_i$(tk.inst)_r$(tk.rep)_p$(tk.p)_$(tk.model)$(tk.scale).jls"), r) catch; end
        wlog(@sprintf("    [噪声] %-5s inst%-3d r%d p=%-2d %s×%g  T=%.1fμs  R_a 无噪声 %.4f → 含噪 %.4f (含读出 %.4f)  [%.1f 分钟]",
                      tk.label, tk.inst, tk.rep, tk.p, tk.model, tk.scale, r.T, r.ra0, mean(r.ra), mean(r.raro), r.t / 60), :always)
        r
    end
    for r in RR
        RES[(r.label, r.inst, r.rep, r.p, r.model, r.scale)] = r
    end
    @printf("计算完成, 墙钟 %.1f 分钟\n", (time() - t_nz) / 60)

    # ── 核查: 自写模拟器的无噪声值 vs checkpoint 里存的 R_a ──
    dv = [abs(r.ra0 - r.ra_stored) for r in values(RES)]
    mprintln(@sprintf("无噪声核查: %d 个任务, 自写模拟器 vs checkpoint 存的 R_a 最大差 %.1e %s", length(dv), maximum(dv),
                      maximum(dv) < 1e-5 ? "✓" : "⚠ (不一致: 检查 ε / mixer 符号 / Ωt 是否与源目录一致)"))
    # 不一致的任务: 用 Bloqade 本身(loss_fn_*)再算一遍同一组角度, 判断是"自写模拟器错"还是"checkpoint 里角度与存的值对不上"
    BADK = sort(unique([(r.label, r.inst, r.rep, r.p) for r in values(RES) if abs(r.ra0 - r.ra_stored) > 1e-5]))
    if !isempty(BADK)
        mprintln("  不一致的 (源, 实例, 重复, 深度): 用 Bloqade 复算同一组角度 ↓")
        for (lab, inst, rep, p) in BADK[1:min(end, 12)]
            t = first(tt for tt in TASKS if tt.label == lab && tt.inst == inst && tt.rep == rep && tt.p == p)
            g = GBY[inst]; r = RES[(lab, inst, rep, p, "full", 1.0)]
            bq = try
                t.cond == "cold_start" ? -loss_fn_weighted(t.x, g.atoms, g.edges, g.weights, g.n_atoms) / g.cmax :
                    -loss_fn_warmstart_weighted(t.x, clamp.(t.pg, max(1e-6, t.eps), 1.0 - max(1e-6, t.eps)), g.atoms, g.edges,
                                                g.weights, g.n_atoms; mixsign=t.mixsign) / g.cmax
            catch e
                NaN
            end
            mprintln(@sprintf("    %s inst%d r%d p=%d: 存的 %.4f | 自写 %.4f | Bloqade %.4f  (总时长 %.2f μs)", lab, inst, rep, p,
                              r.ra_stored, r.ra0, bq, r.T))
            mprintln("      角度 [γ1,β1,γ2,β2,...] = " * join([@sprintf("%.4f", v) for v in t.x], ", "))
            # 变体: 去掉末尾的零层 / 最后一个非零层只留代价段 / 只留混合段 / 代价段改成 1 μs —— 两个模拟器各算一遍
            pl = length(t.x) ÷ 2; L = findlast(k -> t.x[2k-1] + t.x[2k] > 0, 1:pl); L === nothing && (L = pl)
            xt = t.x[1:2L]
            vars = [("去掉末尾零层", xt), ("第$(L)层 β=0", (y = copy(xt); y[2L] = 0.0; y)),
                    ("第$(L)层 γ=0", (y = copy(xt); y[2L-1] = 0.0; y)), ("第$(L)层 γ=1", (y = copy(xt); y[2L-1] = 1.0; y)),
                    ("只到第$(L-1)层", xt[1:2L-2])]
            pos = nz_pos(g); NB = nz_bits(g.n_atoms); Nb = vec(sum(NB, dims=2)); E0 = nz_eint(NB, nz_vmat(pos))
            for (nm, xv) in vars
                isempty(xv) && continue
                sch = nz_schedule(xv, t.cond, pos, t.Ωt; pg=t.pg, eps=t.eps, mixsign=t.mixsign)
                mine = nz_cut(nz_traj(sch, E0, NB, Nb, g.n_atoms, 0.0, 0.0, MersenneTwister(0))[1], NB, g.edges, g.weights, 0.0, 0.0)[1] / g.cmax
                bqv = try
                    t.cond == "cold_start" ? -loss_fn_weighted(xv, g.atoms, g.edges, g.weights, g.n_atoms) / g.cmax :
                        -loss_fn_warmstart_weighted(xv, clamp.(t.pg, max(1e-6, t.eps), 1.0 - max(1e-6, t.eps)), g.atoms, g.edges,
                                                    g.weights, g.n_atoms; mixsign=t.mixsign) / g.cmax
                catch e
                    NaN
                end
                mprintln(@sprintf("      变体 %-14s 自写 %.6f | Bloqade %.6f", nm, mine, bqv))
            end
        end
    end

    nz_se(v) = length(v) > 1 ? std(v) / sqrt(length(v)) : 0.0
    nz_sign_p(w, l) = (n = w + l; n == 0 ? 1.0 : min(1.0, 2 * sum(Float64(binomial(BigInt(n), i)) for i in 0:min(w, l)) / 2.0^n))
    LABS = [s.label for s in NZ_SRC]
    # 每实例(对重复取平均)的值
    function nz_inst(lab, p, mdl, λ, fld)
        out = Dict{Int,Float64}()
        for g in GRAPHS
            v = [getfield(RES[k], fld) isa AbstractVector ? mean(getfield(RES[k], fld)) : getfield(RES[k], fld)
                 for k in keys(RES) if k[1] == lab && k[2] == g.inst && k[4] == p && k[5] == mdl && k[6] == λ]
            isempty(v) || (out[g.inst] = mean(v))
        end
        return out
    end
    PS(lab) = sort(unique([k[4] for k in keys(RES) if k[1] == lab]))

    mprintln("\n" * "="^78)
    mprintln("【表N1】Aquila 噪声(λ=1, full)下的 R_a: 每实例先对轨迹/重复取平均, 再对实例平均±标准误")
    mprintln("       无噪声 = 原值; 含噪 = 相干+非相干(理想测量); +读出 = 再计入读出误差(硬件实际测得的)")
    mprintln("="^78)
    for lab in LABS
        mprintln("  [$(lab)]   p   总时长μs   无噪声            含噪               含噪+读出          噪声损失")
        for p in PS(lab)
            a0 = nz_inst(lab, p, "full", 1.0, :ra0); a1 = nz_inst(lab, p, "full", 1.0, :ra); a2 = nz_inst(lab, p, "full", 1.0, :raro)
            T = nz_inst(lab, p, "full", 1.0, :T)
            isempty(a0) && continue
            v0 = collect(values(a0)); v1 = [a1[i] for i in keys(a0)]; v2 = [a2[i] for i in keys(a0)]
            mprintln(@sprintf("          %3d  %7.2f   %.4f±%.4f    %.4f±%.4f     %.4f±%.4f    %+.4f",
                              p, mean(values(T)), mean(v0), nz_se(v0), mean(v1), nz_se(v1), mean(v2), nz_se(v2), mean(v2) - mean(v0)))
        end
    end

    mprintln("\n" * "="^78)
    mprintln("【表N2】噪声来源分解(λ=1): 各来源单独开时的损失(相对无噪声, R_a 单位; 负 = 变差)")
    mprintln("       coh = 准静态 Ω/Δ/位置涨落; inc = 退相位+弛豫; 读出 = 只加读出误差(无噪声末态); full = 全部(不含读出)")
    mprintln("="^78)
    for lab in LABS
        mprintln("  [$(lab)]   p    coh        inc        读出       full       full+读出")
        for p in PS(lab)
            a0 = nz_inst(lab, p, "full", 1.0, :ra0); isempty(a0) && continue
            m0 = mean(values(a0))
            f(mdl, fld) = (d = nz_inst(lab, p, mdl, 1.0, fld); isempty(d) ? NaN : mean(values(d)) - m0)
            mprintln(@sprintf("          %3d  %+.4f   %+.4f   %+.4f   %+.4f   %+.4f", p, f("coh", :ra), f("inc", :ra),
                              f("full", :ra0ro), f("full", :ra), f("full", :raro)))
        end
    end

    LAMS = sort(unique([λ for (m, λ) in NZ_MODELS if m == "full"]))
    mprintln("\n" * "="^78)
    mprintln("【表N3】噪声强度扫描 (full, 含读出): R_a vs 深度 p; λ = 噪声强度 / Aquila;  ★ = 该 λ 下最好的深度")
    mprintln("="^78)
    BESTP = Dict{Tuple{String,Float64},Int}()
    for lab in LABS
        mprintln("  [$(lab)]   p   " * join([@sprintf("λ=%-9g", λ) for λ in LAMS], "  ") * "  无噪声")
        cols = Dict(λ => Dict(p => (d = nz_inst(lab, p, "full", λ, :raro); isempty(d) ? NaN : mean(values(d))) for p in PS(lab)) for λ in LAMS)
        for λ in LAMS
            vals = [(cols[λ][p], p) for p in PS(lab) if !isnan(cols[λ][p])]
            isempty(vals) || (BESTP[(lab, λ)] = maximum(vals)[2])
        end
        for p in PS(lab)
            a0 = nz_inst(lab, p, "full", 1.0, :ra0)
            mprintln(@sprintf("          %3d  ", p) * join([@sprintf("%.4f%s     ", cols[λ][p], get(BESTP, (lab, λ), 0) == p ? "★" : " ") for λ in LAMS], "  ") *
                     @sprintf("  %.4f", isempty(a0) ? NaN : mean(values(a0))))
        end
    end

    if length(LABS) >= 2
        mprintln("\n" * "="^78)
        mprintln("【表N4】两条线在噪声下的对比与组合(full, 含读出; 每条线取该 λ 下平均最好的深度 ★)")
        mprintln("       组合 = 每个实例取两条线中测得 ⟨C⟩ 更大者(含噪测得值, 不需要知道 C_max)")
        mprintln("="^78)
        la, lb = LABS[1], LABS[2]
        for λ in LAMS
            (haskey(BESTP, (la, λ)) && haskey(BESTP, (lb, λ))) || continue
            pa, pb = BESTP[(la, λ)], BESTP[(lb, λ)]
            A = nz_inst(la, pa, "full", λ, :raro); B = nz_inst(lb, pb, "full", λ, :raro)
            ks = sort([i for i in keys(A) if haskey(B, i)]); isempty(ks) && continue
            va = [A[i] for i in ks]; vb = [B[i] for i in ks]; vp = max.(va, vb)
            w = count(va .> vb .+ 1e-4); l = count(vb .> va .+ 1e-4)
            mprintln(@sprintf("  λ=%-4g  %s(p=%d) %.4f   %s(p=%d) %.4f   %s 好 %d / %s 好 %d (符号检验 p=%.3f)   组合 %.4f (选 %s %d 次)",
                              λ, la, pa, mean(va), lb, pb, mean(vb), la, w, lb, l, nz_sign_p(w, l), mean(vp), la, count(va .>= vb)))
        end
    end

    # ── 采样指标 ──
    nzv(lab, p, λ, f) = (v = [f(RES[k]) for k in keys(RES) if k[1] == lab && k[4] == p && k[5] == "full" && k[6] == λ && hasproperty(RES[k], :popt)];
                         isempty(v) ? NaN : mean(v))
    mprintln("\n" * "="^78)
    mprintln("【表N5】采样指标(λ=1, full, 含读出; 硬件上的实际用法: 采样 N 次取最好的割)")
    mprintln("       P_opt = 一次采样就得到最优割的概率(均匀随机猜 = 最优解个数/4096); 最好/C_max = N 次采样中最好割的期望")
    mprintln("="^78)
    for lab in LABS
        mprintln("  [$(lab)]   p   P_opt 含噪   (无噪声)   (随机)    最好@10   最好@100  最好@1000 | 随机@100  随机@1000")
        for p in PS(lab)
            isnan(nzv(lab, p, 1.0, r -> r.popt)) && continue
            mprintln(@sprintf("          %3d  %.4f     (%.4f)   (%.4f)   %.4f    %.4f    %.4f   |  %.4f    %.4f", p,
                              nzv(lab, p, 1.0, r -> r.popt), nzv(lab, p, 1.0, r -> r.popt0), nzv(lab, p, 1.0, r -> r.poptu),
                              nzv(lab, p, 1.0, r -> r.best[1]), nzv(lab, p, 1.0, r -> r.best[2]), nzv(lab, p, 1.0, r -> r.best[3]),
                              nzv(lab, p, 1.0, r -> r.bestu[2]), nzv(lab, p, 1.0, r -> r.bestu[3])))
        end
    end
    mprintln("\n" * "="^78)
    mprintln("【表N6】各噪声强度下: 每条线 P_opt 最大的深度, 及其 P_opt / 最好@100 / 最好@1000 (full, 含读出)")
    mprintln("="^78)
    for λ in LAMS
        line = @sprintf("  λ=%-4g", λ)
        for lab in LABS
            cand = [(nzv(lab, p, λ, r -> r.popt), p) for p in PS(lab)]
            cand = filter(c -> !isnan(c[1]), cand); isempty(cand) && continue
            pv, pb = maximum(cand)
            line *= @sprintf("   %s p=%-2d P_opt %.4f  @100 %.4f  @1000 %.4f", lab, pb, pv, nzv(lab, pb, λ, r -> r.best[2]), nzv(lab, pb, λ, r -> r.best[3]))
        end
        lab1 = LABS[1]; pp = PS(lab1)
        isempty(pp) || (line *= @sprintf("   | 随机 P_opt %.4f @100 %.4f @1000 %.4f", nzv(lab1, pp[1], λ, r -> r.poptu),
                                         nzv(lab1, pp[1], λ, r -> r.bestu[2]), nzv(lab1, pp[1], λ, r -> r.bestu[3])))
        mprintln(line)
    end

    # ── CSV ──
    csv = joinpath(directoryBasis, "noise_eval_n$(n_at)_T$(NZ_NTRAJ)_s$(NZ_SEED)$(ptag)_I$(INST_SETFP).csv")
    open(csv, "w") do io
        println(io, "label,cond,inst,rep,p,model,scale,ntraj,T_us,ra_stored,ra_ideal,ra_ideal_ro,ra_noisy,ra_noisy_se,ra_noisy_ro,ra_noisy_ro_se,mean_jumps," *
                    "popt_ro,best10_ro,best100_ro,best1000_ro,popt_ideal,best100_ideal,popt_uniform,best100_uniform,best1000_uniform")
        for k in sort(collect(keys(RES)))
            r = RES[k]
            hs = hasproperty(r, :popt)
            sv = hs ? (r.popt, r.best[1], r.best[2], r.best[3], r.popt0, r.best0[2], r.poptu, r.bestu[2], r.bestu[3]) : ntuple(_ -> NaN, 9)
            @printf(io, "%s,%s,%d,%d,%d,%s,%g,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.label, r.cond, r.inst, r.rep, r.p, r.model, r.scale, r.ntraj, r.T, r.ra_stored, r.ra0, r.ra0ro,
                    mean(r.ra), nz_se(r.ra), mean(r.raro), nz_se(r.raro), r.nj, sv...)
        end
    end
    mprintln("✓ 逐任务CSV → $(basename(csv))")
end
if length(ARGS) >= 1 && ARGS[1] == "noise_eval"
    nz_eval_main()
    exit(0)
end

if length(ARGS) >= 1 && ARGS[1] == "omega_scan"
    using Statistics
    n_at        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    n_instances = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
    OM_LIST     = length(ARGS) >= 4 ? parse.(Float64, split(ARGS[4], ",")) : [1.0, 2.0, 4.0, 8.0, 15.7]
    n_grid      = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 60
    2.0 in OM_LIST || push!(OM_LIST, 2.0)
    OM_LIST = sort(unique(OM_LIST))
    all(>(0), OM_LIST) || error("Ωt 必须为正")
    OM_CONDS = String.(split(get(ENV, "QAOA_TL_CONDS", "cold_start,gnn_warmstart"), ","))
    optimizer = "zygote"
    println("\n" * "="^70)
    println("Ωt 扫描 (方法二, p=1 精确最优: $(n_grid)×$(n_grid) 网格 + NM 精修)")
    println("n=$(n_at)  实例数=$(n_instances)  Ωt ∈ {$(join(OM_LIST, ", "))}  条件: $(join(OM_CONDS, ", "))")
    println("每个 Ωt: γ ∈ [0, $(round(3π*QAOA_BOX_SCALE, digits=3))],  β ∈ [0, π·$(QAOA_BOX_SCALE)·2/Ωt] (转角 Ωt·β ∈ [0, 2π·$(QAOA_BOX_SCALE)])")
    println("硬件参照: Ω0 = 5π ≈ 15.7 rad/μs 已用于态制备; Ωt 超过它的值仅供理论参考")
    println("="^70)
    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" *
                 (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    isfile(prior_file) || error("找不到先验 $(prior_tag)")
    gnn_params = deserialize(prior_file)
    GRAPHS, INST_FPS, INST_SETFP, INST_FROM_FILE =
        obtain_test_instances(n_at, n_instances, gnn_params; verbose=false, save=true, prior_tag=prior_tag)
    boxtag = QAOA_BOX_SCALE == 1.0 ? "" : "_box$(QAOA_BOX_SCALE)"
    om_dir = joinpath(directoryBasis, "ckpt_omscan_n$(n_at)_G$(n_grid)" * boxtag)
    mkpath(om_dir)
    check_ckpt_fingerprints(om_dir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE)
    om_fname(Ωt, inst, c) = "om_i$(inst)_$(c)_W$(Ωt).jls"
    OMR = Dict{Tuple{Float64,Int,String},Any}(); TODO = NamedTuple[]
    for Ωt in OM_LIST, g in GRAPHS, c in OM_CONDS
        f = joinpath(om_dir, om_fname(Ωt, g.inst, c))
        r = isfile(f) ? (try deserialize(f) catch; nothing end) : nothing
        r === nothing ? push!(TODO, (g=g, cond=c, Ωt=Ωt, n_grid=n_grid)) : (OMR[(Ωt, g.inst, c)] = r)
    end
    @printf("任务: 共 %d 个, 已缓存 %d, 本次计算 %d (worker=%d)\n", length(TODO) + length(OMR), length(OMR), length(TODO), nworkers())
    t_om = time(); odir = om_dir
    RR = isempty(TODO) ? Any[] : pmap(TODO) do tk
        r = om_task(tk)
        try serialize(joinpath(odir, "om_i$(tk.g.inst)_$(tk.cond)_W$(tk.Ωt).jls"), r) catch; end
        wlog(@sprintf("    [Ωt 扫描] Ωt=%-5g inst%-3d %-13s R_a(p=1)=%.4f  γ=%.3f β=%.3f  [%.1f 分钟]",
                      tk.Ωt, tk.g.inst, tk.cond, r.ra, r.x[1], r.x[2], r.t/60), :always)
        r
    end
    for r in RR
        OMR[(r.Ωt, r.inst, r.cond)] = r
    end
    @printf("计算完成, 墙钟 %.1f 分钟\n", (time() - t_om)/60)

    om_sign_p(w, l) = (n = w + l; n == 0 ? 1.0 :
                       min(1.0, 2*sum(Float64(binomial(BigInt(n), i)) for i in 0:min(w, l)) / 2.0^n))
    om_se(v) = length(v) > 1 ? std(v)/sqrt(length(v)) : 0.0

    # Ωt=2 核对: 与核消融 / transfer_bo 缓存的 p=1 网格值比较(同一算法, 应几乎相同)
    if n_grid == 60
        CHK = Dict{Tuple{Int,String},Float64}()
        for d in filter(x -> startswith(x, "ckpt_abl_n$(n_at)_") && occursin("_G60", x) &&
                             (QAOA_BOX_SCALE == 1.0 ? !occursin("_box", x) : occursin(boxtag, x)), readdir(directoryBasis))
            dd = joinpath(directoryBasis, d)
            ok = try check_ckpt_fingerprints(dd, GRAPHS, INST_FPS; from_file=INST_FROM_FILE, write=false); true catch; false end
            ok || continue
            for g in GRAPHS, c in OM_CONDS
                f = joinpath(dd, "t_i$(g.inst)_$(c)_p1_grid_r0.jls")
                (haskey(CHK, (g.inst, c)) || !isfile(f)) && continue
                r = try deserialize(f) catch; nothing end
                r === nothing || (CHK[(g.inst, c)] = -r.ybest / g.cmax)
            end
        end
        for d in filter(x -> startswith(x, "ckpt_tlsrc_n$(n_at)_") && !occursin("_ws", x) && !occursin("_om", x) &&
                             (QAOA_BOX_SCALE == 1.0 ? !occursin("_box", x) : occursin(boxtag, x)), readdir(directoryBasis))
            for g in GRAPHS, c in OM_CONDS
                f = joinpath(directoryBasis, d, "grid_i$(g.inst)_$(c).jls")
                (haskey(CHK, (g.inst, c)) || !isfile(f)) && continue
                r = try deserialize(f) catch; nothing end
                r === nothing || (CHK[(g.inst, c)] = r)
            end
        end
        dv = [abs(OMR[(2.0, i, c)].ra - v) for ((i, c), v) in CHK if haskey(OMR, (2.0, i, c))]
        println(isempty(dv) ? "Ωt=2 核对: 没找到旧的 p=1 网格缓存, 跳过" :
                @sprintf("Ωt=2 核对: 与旧的 p=1 网格标准答案比较 %d 个, 最大差 %.2e %s", length(dv), maximum(dv),
                         maximum(dv) < 1e-4 ? "(一致 ✓)" : "(⚠ 不一致: 检查 Ωt / β 盒子的改动)"))
    end

    mprintln("\n" * "="^78)
    mprintln("【表Ω1】p=1 精确最优 R_a 随混合层幅度 Ωt 的变化 (20 个实例, 均值±标准误)")
    mprintln("      Δ = R_a(Ωt) − R_a(Ωt=2), 按实例配对; 赢/输 = Δ>1e-4 / Δ<−1e-4 的实例数")
    mprintln("      最优 = 该 Ωt 在多少个实例上是所有 Ωt 里最好的(并列都算); β贴边 = 最优点的 β 贴住盒子上界(盒子可能太小)")
    mprintln("      转角 = Ωt·β* (混合层的名义转角, rad)")
    mprintln("="^78)
    for c in OM_CONDS
        mprintln("\n  $(c)")
        mprintln("  " * rpad("Ωt", 7) * rpad("R_a(p=1)", 18) * rpad("中位数", 9) * rpad("Δ vs Ωt=2", 19) *
                 rpad("赢/输", 9) * rpad("p_sign", 9) * rpad("最优", 6) * rpad("β贴边", 7) * "转角均值")
        best = Dict(g.inst => maximum(OMR[(Ωt, g.inst, c)].ra for Ωt in OM_LIST) for g in GRAPHS)
        for Ωt in OM_LIST
            v = [OMR[(Ωt, g.inst, c)].ra for g in GRAPHS]
            d = [OMR[(Ωt, g.inst, c)].ra - OMR[(2.0, g.inst, c)].ra for g in GRAPHS]
            w = count(>(1e-4), d); l = count(<(-1e-4), d)
            nb = count(g -> OMR[(Ωt, g.inst, c)].ra >= best[g.inst] - 1e-6, GRAPHS)
            ne = count(g -> OMR[(Ωt, g.inst, c)].edge_b, GRAPHS)
            rot = mean(Ωt * OMR[(Ωt, g.inst, c)].x[2] for g in GRAPHS)
            mprintln("  " * rpad(@sprintf("%g", Ωt), 7) * rpad(@sprintf("%.4f±%.4f", mean(v), om_se(v)), 18) *
                     rpad(@sprintf("%.4f", median(v)), 9) * rpad(@sprintf("%+.4f±%.4f", mean(d), om_se(d)), 19) *
                     rpad("$(w)/$(l)", 9) * rpad(@sprintf("%.3g", om_sign_p(w, l)), 9) * rpad("$(nb)", 6) *
                     rpad("$(ne)", 7) * @sprintf("%.2f", rot))
        end
    end
    mprintln("\n" * "="^78)
    mprintln("【表Ω2】逐实例 p=1 精确最优 R_a (★ = 该实例最好的 Ωt)")
    mprintln("="^78)
    for c in OM_CONDS
        mprintln("\n  $(c)")
        mprintln("  " * rpad("实例", 6) * join(rpad(@sprintf("Ωt=%g", Ωt), 11) for Ωt in OM_LIST))
        for g in GRAPHS
            vs = [OMR[(Ωt, g.inst, c)].ra for Ωt in OM_LIST]; bm = maximum(vs)
            mprintln("  " * rpad("i$(g.inst)", 6) * join(rpad(@sprintf("%.4f%s", x, x >= bm - 1e-6 ? "★" : " "), 11) for x in vs))
        end
    end
    tagc = "n$(n_at)_G$(n_grid)" * boxtag * (INST_FROM_FILE ? "_I$(INST_SETFP)" : "")
    fc = joinpath(directoryBasis, "omega_scan_$(tagc).csv")
    open(fc, "w") do io
        println(io, "omega_t,instance,condition,ra_p1,gamma,beta,rotation,beta_at_edge,gamma_at_edge")
        for Ωt in OM_LIST, g in GRAPHS, c in OM_CONDS
            r = OMR[(Ωt, g.inst, c)]
            @printf(io, "%g,%d,%s,%.6f,%.6f,%.6f,%.6f,%d,%d\n", Ωt, g.inst, c, r.ra, r.x[1], r.x[2], Ωt*r.x[2], r.edge_b, r.edge_g)
        end
    end
    mprintln("\n✓ Ωt 扫描 CSV → $(basename(fc))")
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── landscape_scan: 用**生产代码本身**扫描逐层扩展的能量曲面 ───────────────
# 目的: 之前论文里的 landscape 图(fig_landscape)是另一套 Python 重写的模拟器
# 算的, 其中一个检查点(p=1 最优值)与本代码的网格审计对不上。这里直接调用
# 生产用的 loss_fn_weighted / loss_fn_warmstart_weighted, 在冻结的实例集上
# 重新扫描, 保证图和分析用的就是我们真实优化的那个曲面。
#
# 扫描方式与 fig_landscape 相同(逐层扩展): 第 p 层时前 p-1 层固定为上一步
# 找到的最优, 只在新一层 (γ_p, β_p) ∈ [0,3π]×[0,π] 上做 n_grid×n_grid 网格,
# 再从网格前 3 个点做二维 NM 精修, 得到该层最优后进入下一层。
# 自检: (γ_p,β_p)=(0,0) 是恒等层, 其值必须等于上一层最优(到 1e-9)。
# 频谱诊断(每层, 去均值 + Hann 窗后的二维 FFT):
#   · Fourier 密度 s = ‖c‖₁²/‖c‖₂²  (等效"有效频率数", 与旧论文表 II 同定义)
#   · 前 8 / 14 个频点(对称成对, 即 4 / 7 个独立频率)占总功率的比例
#     —— 4 = 我们的 2G+2C 成分数, 7 = 原论文 6C+1G。比例低 ⇒ 少数几个谱峰
#     描述不了这个曲面, 谱混合核的先天假设不成立。
# 运行方式:
#   julia r3g_sge_integrated.jl landscape_scan <n_atoms> <实例列表> <p_max> <n_grid> [条件列表]
# 例如:  ... landscape_scan 12 1,6 5 60 cold_start,gnn_warmstart
# ══════════════════════════════════════════════════════════════════════════
@everywhere function ls_task(tk)
    g = tk.g; cnd = tk.cond; n_grid = tk.n_grid
    evf = x -> begin
        lo = zeros(length(x)); hi = vcat([[BOX_HI[1], BOX_HI[2]] for _ in 1:(length(x) ÷ 2)]...)
        xq = clamp.(x, lo, hi)
        try
            lv = cnd == "gnn_warmstart" ?
                 loss_fn_warmstart_weighted(xq, g.best_p_gnn, g.atoms, g.edges, g.weights, g.n_atoms) :
                 loss_fn_weighted(xq, g.atoms, g.edges, g.weights, g.n_atoms)
            isfinite(lv) ? lv : NaN
        catch
            NaN
        end
    end
    γs = collect(range(0.0, BOX_HI[1], length=n_grid)); βs = collect(range(0.0, BOX_HI[2], length=n_grid))
    prefix = Float64[]; layers = NamedTuple[]
    for p in 1:tk.p_max
        t0 = time()
        G = [-evf(vcat(prefix, [γ, β])) / g.cmax for γ in γs, β in βs]      # R_a, 行=γ 列=β
        # 二维 NM 精修(只动新一层)
        idx = sortperm(vec(G), rev=true)
        starts = Vector{Vector{Float64}}()
        for k in idx
            i, j = Tuple(CartesianIndices(G)[k]); x = [γs[i], βs[j]]
            all(s0 -> maximum(abs.(s0 .- x) ./ [BOX_HI[1], BOX_HI[2]]) > 0.05, starts) && push!(starts, x)
            length(starts) >= 3 && break
        end
        bu = starts[1]; bv = G[idx[1]]
        for x0 in starts
            r = optimize(u -> evf(vcat(prefix, clamp.(u, [0.0, 0.0], [BOX_HI[1], BOX_HI[2]]))), x0, NelderMead(),
                         Optim.Options(iterations=400, f_reltol=1e-12))
            v = -Optim.minimum(r) / g.cmax
            v > bv && (bv = v; bu = clamp.(Optim.minimizer(r), [0.0, 0.0], [BOX_HI[1], BOX_HI[2]]))
        end
        push!(layers, (p=p, grid=G, best_u=bu, best_ar=bv, origin=G[1, 1], t=time()-t0))
        wlog(@sprintf("    [landscape] inst%d %-13s p=%d  R_a∈[%.4f, %.4f]  原点=%.6f  最优(精修)=%.6f  [%.1f 分钟]",
                      g.inst, cnd, p, minimum(G), maximum(G), G[1, 1], bv, (time()-t0)/60), :always)
        prefix = vcat(prefix, bu)
    end
    return (inst=g.inst, seed=g.seed, cond=cnd, γs=γs, βs=βs, layers=layers,
            unwanted=g.unwanted, cmax_over_w=g.cmax/g.total_w)
end

if length(ARGS) >= 1 && ARGS[1] == "landscape_scan"
    using Statistics, LinearAlgebra
    n_at     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    inst_sel = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [1, 6]
    p_max    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
    n_grid   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 60
    conds    = length(ARGS) >= 6 ? String.(split(ARGS[6], ",")) : ["cold_start", "gnn_warmstart"]
    optimizer = "zygote"
    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" *
                 (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    isfile(prior_file) || error("找不到先验 $(prior_tag)")
    GRAPHS, INST_FPS, INST_SETFP, INST_FROM_FILE =
        obtain_test_instances(n_at, maximum(inst_sel), deserialize(prior_file); verbose=false, save=true, prior_tag=prior_tag)
    println("\n" * "="^70)
    println("能量曲面扫描(生产代码, 逐层扩展)  实例=$(inst_sel)  p≤$(p_max)  网格 $(n_grid)×$(n_grid)  条件=$(conds)")
    println("参数盒每层 [0,$(round(BOX_HI[1],digits=3))]×[0,$(round(BOX_HI[2],digits=3))]")
    println("="^70)
    TK = [(g=GRAPHS[i], cond=c, p_max=p_max, n_grid=n_grid) for i in inst_sel for c in conds]
    LS = pmap(ls_task, TK)

    # ── 频谱诊断 ──
    function spec_stats(G::Matrix{Float64})
        A = G .- mean(G)
        w1 = [0.5 - 0.5cos(2π*(i-1)/(size(A,1)-1)) for i in 1:size(A,1)]
        w2 = [0.5 - 0.5cos(2π*(j-1)/(size(A,2)-1)) for j in 1:size(A,2)]
        F = abs.(fft(A .* (w1 * w2')))
        F[1, 1] = 0.0
        c = vec(F); P = sort(c.^2, rev=true); tot = sum(P)
        s_dens = sum(c)^2 / max(sum(c.^2), 1e-300)
        return (s=s_dens, top8=sum(P[1:min(8,end)])/tot, top14=sum(P[1:min(14,end)])/tot)
    end
    mprintln("\n【表L】逐层扩展曲面的统计 (R_a 单位)")
    mprintln("  实例(seed) 条件          p | R_a 范围           动态范围  原点自检   最优(精修)  新层最大提升  提升>0面积 | Fourier密度  前4频率功率  前7频率功率")
    rows = NamedTuple[]
    for r in LS
        prev = NaN
        for L in r.layers
            st = spec_stats(L.grid)
            dr = maximum(L.grid) - minimum(L.grid)
            ok = isnan(prev) ? "—" : (abs(L.origin - prev) < 1e-9 ? "✓" : @sprintf("✗%.1e", L.origin - prev))
            gain = isnan(prev) ? L.best_ar - L.origin : L.best_ar - prev
            fpos = isnan(prev) ? count(>(L.origin), L.grid)/length(L.grid) : count(>(prev + 1e-12), L.grid)/length(L.grid)
            @printf("  %3d(%3d) %-13s %d | [%.4f, %.4f]  %8.4f   %-7s  %.6f   %+.5f     %6.1f%%    | %8.1f      %5.1f%%      %5.1f%%\n",
                    r.inst, r.seed, r.cond, L.p, minimum(L.grid), maximum(L.grid), dr, ok, L.best_ar,
                    gain, 100fpos, st.s, 100st.top8, 100st.top14)
            push!(rows, (inst=r.inst, seed=r.seed, cond=r.cond, p=L.p, rmin=minimum(L.grid), rmax=maximum(L.grid),
                         origin=L.origin, best=L.best_ar, gain=gain, fpos=fpos, s=st.s, top8=st.top8, top14=st.top14))
            prev = L.best_ar
        end
    end
    tagl = "n$(n_at)_G$(n_grid)_p$(p_max)" * (QAOA_BOX_SCALE == 1.0 ? "" : "_box$(QAOA_BOX_SCALE)") *
           (INST_FROM_FILE ? "_I$(INST_SETFP)" : "")
    fg = joinpath(directoryBasis, "landscape_grid_$(tagl).csv")
    open(fg, "w") do io
        println(io, "instance,seed,condition,p,gamma,beta,ar_cmax")
        for r in LS, L in r.layers, (i, γ) in enumerate(r.γs), (j, β) in enumerate(r.βs)
            @printf(io, "%d,%d,%s,%d,%.6f,%.6f,%.8f\n", r.inst, r.seed, r.cond, L.p, γ, β, L.grid[i, j])
        end
    end
    fsum = joinpath(directoryBasis, "landscape_summary_$(tagl).csv")
    open(fsum, "w") do io
        println(io, "instance,seed,condition,p,ar_min,ar_max,origin,best_polished,layer_gain,frac_improving,fourier_density,power_top4,power_top7")
        for x in rows
            @printf(io, "%d,%d,%s,%d,%.6f,%.6f,%.8f,%.8f,%.6f,%.4f,%.2f,%.4f,%.4f\n", x.inst, x.seed, x.cond, x.p,
                    x.rmin, x.rmax, x.origin, x.best, x.gain, x.fpos, x.s, x.top8, x.top14)
        end
    end
    mprintln("\n✓ 网格数据  → $(basename(fg))   (可直接用来重画 landscape 图)")
    mprintln("✓ 统计汇总  → $(basename(fsum))")
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── export_instances: 只导出测试实例集, 不做任何 QAOA 计算 ──────────────────
# 两种来源:
#  (a) 从已有的 batch_qaoa_*.jls 导出 —— **最推荐**: 那里面存的就是当初那次
#      运行实际用过的图(含当时的 GNN 输出), 导出后别的机器能逐比特复现。
#      .jls 必须在生成它的同一台机器 / 同一 Julia 版本上读, 所以这一步要在
#      原机器上跑(比如集群), 导出的 .txt 才是跨平台的。
#  (b) 不给 jls: 按当前机器的规则重新生成(需要已缓存的先验)并导出。
# 运行方式:
#   julia r3g_sge_integrated.jl export_instances <n_atoms> <n_instances> [optimizer] [batch_jls路径]
# 例如(在集群上, 导出那次 342 小时 sweep 运行用的 20 张图):
#   julia --project=ChenQAOA r3g_sge_integrated.jl export_instances 12 20 zygote qaoa_out/batch_qaoa_n12_inst20_zygote.jls
# ══════════════════════════════════════════════════════════════════════════
if length(ARGS) >= 1 && ARGS[1] == "export_instances"
    n_at        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    n_instances = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
    optimizer   = length(ARGS) >= 4 ? ARGS[4] : "zygote"
    src_jls     = length(ARGS) >= 5 ? ARGS[5] : ""
    GR_EXP = if !isempty(src_jls)
        isfile(src_jls) || (println("找不到 $(src_jls)"); exit(1))
        dat = deserialize(src_jls)
        hasproperty(dat, :GRAPHS) || (println("$(src_jls) 里没有 GRAPHS 字段, 不是 batch_qaoa_*.jls"); exit(1))
        gs = collect(dat.GRAPHS)
        gs[1].n_atoms == n_at || (println("jls 里是 n_atoms=$(gs[1].n_atoms), 命令给的是 $(n_at)"); exit(1))
        length(gs) >= n_instances || (println("jls 里只有 $(length(gs)) 个实例"); exit(1))
        println("从 $(basename(src_jls)) 导出前 $(n_instances) 个实例(即当初那次运行实际用的图)")
        [merge(g, (inst=k,)) for (k, g) in enumerate(gs[1:n_instances])]
    else
        ptag = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" *
               (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
        pf = joinpath(directoryBasis, ptag)
        isfile(pf) || (println("找不到先验 $(ptag)(生成测试集要用它算 GNN 输入)"); exit(1))
        println("按本机规则重新生成 $(n_instances) 个实例...")
        build_test_instances(n_at, n_instances, deserialize(pf); verbose=true)
    end
    sfp_exp = set_fp(GR_EXP)
    mkpath(directoryBasis)
    out_exp = joinpath(directoryBasis, "instances_n$(n_at)_inst$(length(GR_EXP))_$(sfp_exp).txt")
    save_instance_set(out_exp, GR_EXP; note = isempty(src_jls) ? "source=regenerated" : "source=$(basename(src_jls))")
    # 立刻读回来校验一遍: 读回的指纹必须与写出前一致, 否则说明格式有问题
    GR_BACK = load_instance_set(out_exp, n_at, length(GR_EXP))
    set_fp(GR_BACK) == sfp_exp || (println("✗ 读回校验失败: 指纹 $(set_fp(GR_BACK)) ≠ $(sfp_exp)"); exit(1))
    println("✓ 已导出并读回校验通过 → $(out_exp)")
    println("  实例集指纹: $(sfp_exp)")
    println("  实例种子: ", join((g.seed for g in GR_EXP), ","))
    println("  在另一台机器上: 拷过去, 设 QAOA_INSTANCES=<路径> 再跑 gnn_batch_qaoa_weighted")
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════
# ── ckpt_report: 直接从 checkpoint 目录出全套报表, 不做任何计算 ───────────
# 用途:
#  · 之前跑完但输出被截断/中断没看到最终表格 —— 数据其实都在 checkpoint 里
#  · 做新旧配置的受控对比 —— 同样的图和BO种子, 只有配置不同, 两个目录各出
#    一份报表直接比
#  · 部分完成的任务集也能出表(会明确标注只用了多少个任务, 结论仅供参考)
#
# 测试图是确定性构造的(种子固定), 所以这里重新建一遍图就能跟当初完全一致,
# 只需要几分钟, 不用重跑几天的 QAOA。
#
# 运行方式:
#   julia r3g_sge.jl ckpt_report <ckpt目录名> [n_atoms] [p_max] [n_instances] [n_bo_seeds] [optimizer]
# 例如:
#   julia r3g_sge.jl ckpt_report ckpt_n12_p3_bs5_zygote 12 3 20 5 zygote
#   julia r3g_sge.jl ckpt_report ckpt_n12_p3_bs5_zygote_d6b100_5 12 3 20 5 zygote
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "ckpt_report"
    ck_name     = length(ARGS) >= 2 ? ARGS[2] : ""
    n_at        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12
    p_max       = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 3
    n_instances = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20
    n_bo_seeds  = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 5
    optimizer   = length(ARGS) >= 7 ? ARGS[7] : "zygote"
    wmin_fixed, wmax_fixed = 0.5, 1.0
    using Statistics, LinearAlgebra

    ck_dir = joinpath(directoryBasis, ck_name)
    if isempty(ck_name) || !isdir(ck_dir)
        println("找不到 checkpoint 目录: $(ck_dir)")
        println("现有的目录:")
        for d in filter(x -> startswith(x, "ckpt_"), readdir(directoryBasis))
            println("  $(d)  →  $(length(readdir(joinpath(directoryBasis, d)))) 个任务")
        end
        exit(1)
    end

    println("\n" * "="^70)
    println("从 checkpoint 出报表(不做任何新计算)")
    println("目录: $(ck_name)")
    println("配置: n=$(n_at)  p_max=$(p_max)  实例数=$(n_instances)  BO种子=$(n_bo_seeds)")
    println("="^70)

    # 先验只用来重建 GNN 推理值(图本身的构造不依赖它)
    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean20_$(optimizer)" *
                 (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    isfile(prior_file) || (println("找不到先验 $(prior_tag), 无法重建 GNN 推理值"); exit(1))
    gnn_params = deserialize(prior_file)

    # 重建测试实例: 设了 QAOA_INSTANCES 就读实例集文件, 否则按原逻辑确定性重建。
    # [注意] "确定性"只在同一台机器/同一 Julia 与包版本下成立 —— 实测 Mac 和
    # 集群生成的是两批不同的图。跨机器出报表时请用 QAOA_INSTANCES 指向当初那台
    # 机器存下的 instances_*.txt。
    GRAPHS, INST_FPS, INST_SETFP, INST_FROM_FILE =
        obtain_test_instances(n_at, n_instances, gnn_params; wmin=wmin_fixed, wmax=wmax_fixed,
                              verbose=false, save=false, prior_tag=prior_tag)
    println("重建了 $(length(GRAPHS)) 个实例")
    # 只读校验(不写指纹文件): 目录里有指纹就必须对得上
    check_ckpt_fingerprints(ck_dir, GRAPHS, INST_FPS; from_file=INST_FROM_FILE, write=false)

    # 加载 checkpoint
    RES = NamedTuple[]
    missing_tasks = Tuple{Int,Int}[]
    for g in GRAPHS, s in 1:n_bo_seeds
        f = joinpath(ck_dir, "task_i$(g.inst)_s$(s).jls")
        if isfile(f)
            try
                push!(RES, deserialize(f))
            catch e
                push!(missing_tasks, (g.inst, s))
            end
        else
            push!(missing_tasks, (g.inst, s))
        end
    end
    n_expect = length(GRAPHS)*n_bo_seeds
    @printf("加载 %d/%d 个任务\n", length(RES), n_expect)
    if !isempty(missing_tasks)
        @printf("⚠ 缺失 %d 个任务, 报表只基于已有的部分, 统计结论仅供参考\n", length(missing_tasks))
        println("  缺失的 (实例,种子): $(missing_tasks[1:min(10,end)])" *
                (length(missing_tasks)>10 ? " ..." : ""))
        # 只保留数据齐全的实例, 否则各实例的 best 不是在同等条件下取的
        bad = Set(t[1] for t in missing_tasks)
        GRAPHS = [g for g in GRAPHS if !(g.inst in bad)]
        RES = [r for r in RES if !(r.inst in bad)]
        @printf("  剔除数据不全的实例后, 剩 %d 个实例可用\n", length(GRAPHS))
        isempty(GRAPHS) && (println("没有数据完整的实例, 无法出表"); exit(1))
        # 重新编号, 让 inst 连续(报表里按 inst 索引)
        remap = Dict(g.inst => i for (i,g) in enumerate(GRAPHS))
        GRAPHS = [merge(g, (inst=remap[g.inst],)) for g in GRAPHS]
        RES = [merge(r, (inst=remap[r.inst],)) for r in RES]
    end

    emit_paper_report(GRAPHS, RES, p_max, n_at, n_bo_seeds, optimizer,
                      ck_name * (INST_FROM_FILE ? "_I$(INST_SETFP)" : ""))
    println("\n输出文件名带上了 checkpoint 目录名后缀, 不会覆盖其它配置的结果。")
    exit(0)
end
# ══════════════════════════════════════════════════════════════════════════
# ── gnn_batch_qaoa_weighted: 一条命令跑 N 个不同实例(不同种子的图) ────────
# 为什么需要这个: 之前所有结论都建立在**一张图**(seed=7)上, 只能说"在这张
# 图上 warmstart 有效", 说不了"warmstart 稳定有效"。这个模式在同一份预训练
# 先验下, 对 N 个独立生成的测试图各跑一遍完整的 p=1..p_max 三方对比, 最后
# 按 p 和 condition 聚合出**跨实例**的均值±标准差 —— 这才是能写进论文的
# 统计结论。
#
# 几个设计上的选择, 说明一下:
#  · 先验只训练/加载**一次**, N 个实例共用 —— 这正是"元学习先验"的用法,
#    也顺便保证了实例之间唯一的变量就是图本身。
#  · 测试种子用 1,2,3,... 训练池种子是 n_atoms*1000+s (n=12 时从 12001 起),
#    两者不重叠, 所以测试图对先验来说都是**没见过**的图, 不存在信息泄漏。
#  · 只接受"干净"的测试图(非边最近距离 > 边最远距离), 跟训练池的筛选标准
#    完全一致 —— 不干净的图物理上不成立(哈密顿量被非边耦合严重污染),
#    放进来会污染统计结论。种子会一直往后试直到凑够 N 个干净实例, 实际
#    试了多少个会如实打印出来。
#  · 逐任务的输出关掉(verbose=false), 否则 10 实例 × 5 层 × 3 条件 × 20 种子
#    = 3000 行会把汇总结果淹掉。每个实例只打印一行进度 + 每层一行结果。
#
# 运行方式:
#   julia r3g_sge.jl gnn_batch_qaoa_weighted <n_atoms> [p_max] [n_instances] [n_epochs] [target_clean_per_size] [optimizer]
# 例如:
#   julia r3g_sge.jl gnn_batch_qaoa_weighted 12 3 10 8000 20 zygote
#   (n=12 的图跑 10 个不同实例, 每个实例 p=1..3)
#
# [QNG 受控对比] 第14个参数 = nm / qng / both (第13个参数束宽对 sweep 无效, 占位填 1):
#   冒烟(几分钟, 只验证流程):  ... gnn_batch_qaoa_weighted 12 2 2 8000 20 zygote 1 0.05 0 sweep 300 1 both
#   快速对比(6实例, p≤3):      ... gnn_batch_qaoa_weighted 12 3 6 8000 20 zygote 1 1.0 0 sweep 300 1 both
#   输出: 两份完整论文报表(文件名后缀 _nm / _qng) + 【QNG 受控对比】表R1-R4
#         + refiner_compare_n12_p3_sweep300_bs1.csv
#
# [跨机器用同一批图] 每次生成测试集都会自动存 instances_n<n>_inst<N>_<指纹>.txt。
#   拷到另一台机器后:  QAOA_INSTANCES=<该文件路径>  再跑同样的命令即可。
#   可选: QAOA_INSTANCES_REINFER=1   用本机先验重新推理 p_gnn(默认用文件里冻结的值)
#         QAOA_TRUST_LEGACY_CKPT=1   信任没有指纹的旧 checkpoint 目录(确认是同一批图时)
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "gnn_batch_qaoa_weighted"
    n_at                   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
    p_max                  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    n_instances            = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
    n_epochs               = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 8000
    target_clean_per_size  = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20
    optimizer              = length(ARGS) >= 7 ? ARGS[7] : "zygote"
    n_bo_seeds             = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 5
    # 第9个参数: BO 预算缩放。默认 1.0 = 正式配置。本地想几分钟内验证整条
    # 流程通不通(而不是等几小时), 填 0.05 之类的小值做冒烟测试。
    # [警告] budget_scale<1 时 BO 远未收敛, 出来的近似比**不能当结果用**。
    budget_scale           = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1.0
    # 第10个参数: BO 内部过程打印间隔。0=关闭(默认), N=每 N 次 BO 迭代打印一行
    # (有提升的迭代无论如何都会打印)。想看清楚 BO 怎么工作就设 10~50;
    # 正式出数据时设 0, 否则日志会非常长(每个 condition 几百行 × 50个任务)。
    bo_verbose             = length(ARGS) >= 10 ? parse(Int, ARGS[10]) : 0
    # 第11个参数: 搜索策略
    #   "bo"             = 每个 p 独立跑 GSM+CSM 贝叶斯优化, 搜索维度 2p
    #   "layerwise"(推荐) = 冻结前 p-1 层, 在新层的**二维**空间里跑 GSM+CSM BO
    #   "layerwise_grid" = 同上但新层用穷举网格代替 BO(最快, 但不体现 GSM+CSM)
    #   "beam"           = 束搜索: 每层保留 K 条互不相同的前缀(K=第8个参数),
    #                      对冲"早期层选错"的风险, 天然并行
    #   "sweep"          = 层间轮扫: 加完新层后回头把前面每层依次重新优化
    #                      (块坐标下降), 直接纠正早期层的次优选择
    #   "hybrid"         = 两者叠加。关键在**顺序**: 扩展 K 条束 → 只对头部
    #                      几条做轮扫 → **再**重新排序 → 保留前 K 条。
    #                      为什么不全部轮扫: 会让所有束收敛到同一个坐标下降
    #                      不动点, 束宽退化成 1。为什么轮扫要在排序之前: 一条
    #                      暂时排第3的前缀可能扫完反超第1, 先排序后砍掉等于
    #                      因为"还没被精修过"就淘汰它, 违背对冲的初衷。
    # beam/sweep 都是为了攻击**前缀约束**: 逐层冻结后早期层不能再改, 实测
    # 实例3 的 p=3 穷举新层全部 3600 个角度仍只能到 0.8914(联合BO 有 0.9029)。
    # 两者内部用的都是同一套二维 GSM+CSM BO(optimize_layer_2d), 只是组织方式
    # 不同, 所以结果可以直接比较。
    # 为什么 layerwise 是推荐项: 搜索维度恒为 2 ⇒ GP 超参数只有 20 维,
    # NelderMead 400 次迭代足以收敛; 而直接在 2p 维联合优化时 p=3 的超参数是
    # 52 维、只跑到收敛所需的 15%, GSM+CSM 那个核实际上退化成了噪声 ——
    # 也就是说逐层分解不是抛弃 GSM+CSM, 而是让它第一次真正发挥作用。
    # 同时保证 R_a 对 p 单调不降(初始设计里含 (0,0), 零时长层=恒等算符)。
    search_mode            = length(ARGS) >= 11 ? ARGS[11] : "bo"
    # 第12个参数: layerwise 下是每层的 BO 预算; layerwise_grid 下是网格分辨率
    # 默认 300: 逐层之后每步 GP 只有 20 维超参数、观测点也少, O(N³) 不再是
    # 瓶颈, 加预算的代价基本是线性的 —— 不像联合BO时是四次方。
    layer_param            = length(ARGS) >= 12 ? parse(Int, ARGS[12]) : (search_mode=="layerwise_grid" ? 40 : 300)
    # 第13个参数: 束宽(仅 beam/hybrid 模式生效, sweep/layerwise 忽略)
    # 默认 = n_bo_seeds(向后兼容)。解耦后可以设 n_bo_seeds=1(避免重复任务)
    # 同时 beam_width=5(在单个任务内部维护5条前缀, 对抗前缀约束)。
    beam_width_param       = length(ARGS) >= 13 ? parse(Int, ARGS[13]) : n_bo_seeds
    # 第14个参数: 局部精修器 (仅 beam/sweep/hybrid 生效)
    #   "nm"   = NelderMead(默认, 原方案, 结果与旧版逐比特一致, checkpoint 目录名也不变)
    #   "qng"  = 量子自然梯度 (Stokes et al. 2020), checkpoint 目录名加 QNG 配置后缀
    #   "both" = 两种都跑并出受控对比报表。NM 部分跟 "nm" 共用 checkpoint 目录,
    #            所以之前用同一配置跑完的 NM 结果会被直接复用, 只需要补算 QNG。
    refiner_param          = length(ARGS) >= 14 ? lowercase(ARGS[14]) : "nm"
    refiner_param in ("nm", "qng", "both") ||
        error("第14个参数(精修器)只能是 nm / qng / both, 收到: $(refiner_param)")
    if refiner_param != "nm" && !(search_mode in ("beam", "sweep", "hybrid"))
        error("精修器 $(refiner_param) 只在 beam/sweep/hybrid 策略下实现, 当前策略是 $(search_mode)。" *
              " 请把第11个参数改成 sweep(推荐)/beam/hybrid, 或第14个参数用 nm。")
    end
    RFS = refiner_param == "both" ? ["nm", "qng"] : [refiner_param]
    wmin_fixed, wmax_fixed = 0.5, 1.0
    using Statistics, LinearAlgebra
    if optimizer == "zygote" && !ensure_zygote()
        println("  → 退回 SPSA 优化器继续"); optimizer = "spsa"
    end

    println("\n" * "="^70)
    println("批量多实例 QAOA warmstart 对比 (target3d)")
    println("n_atoms=$(n_at)  p_max=$(p_max)  实例数=$(n_instances)  优化器=$(optimizer)")
    println("搜索策略=$(search_mode)" *
            (search_mode=="layerwise"      ? "  (逐层冻结 + 新层二维 GSM+CSM BO, 每层预算 $(layer_param))" :
             search_mode=="layerwise_grid" ? "  (逐层冻结 + 新层 $(layer_param)×$(layer_param) 穷举网格)" :
             search_mode=="beam"  ? "  (束搜索: 每层保留 $(beam_width_param) 条互不相同的前缀, 二维 GSM+CSM BO 预算 $(layer_param))" :
             search_mode=="sweep" ? "  (层间轮扫: 新层 BO 预算 $(layer_param) + 回头重优化前面各层)" :
             search_mode=="hybrid" ? "  (束+轮扫叠加: $(beam_width_param) 条前缀, 头部2条做轮扫后再排序)" :
                                             "  (每层独立 GSM+CSM BO, 搜索维度 2p)"))
    println("每个实例内部重启种子数=$(n_bo_seeds)")
    println("GP 版本=$(GP_VERSION)" * (GP_VERSION >= 2 ?
            "  (y标准化 + 输入归一化 + 联合TS + 超参数拟合 $(GP_V2_HP_ITERS) 次迭代/重启+$(GP_V2_EXTRA_RESTARTS); 结果与 v1 基准不可逐比特对照)" :
            "  (原版, 与已有基准逐比特一致)"))
    println("局部精修器=$(refiner_param)" *
            (refiner_param == "nm"   ? "  (NelderMead, 原方案)" :
             refiner_param == "qng"  ? "  (量子自然梯度, $(QNG_CFG_TAG))" :
                                       "  (NM 与 QNG 各跑一遍, 受控对比; QNG 配置 $(QNG_CFG_TAG))"))
    if search_mode in ("beam", "hybrid")
        println("  束宽=$(beam_width_param)" *
                (beam_width_param != n_bo_seeds ? "  (第13个参数显式指定, 与任务重复数 $(n_bo_seeds) 解耦)" : "  (=第8个参数, 未显式指定第13个参数)"))
        if n_bo_seeds > 1
            println("  [提示] 二维BO是确定性的, 多个种子会重复同样的计算, 建议第8个参数设成1,")
            println("         用第13个参数控制束宽(如: … 1 1.0 0 hybrid 300 5)。")
        end
    end
    if search_mode == "sweep" && n_bo_seeds > 1
        println("  提示: sweep 是确定性的(二维BO已验证精确), 多个种子会重复同样的计算,")
        println("        建议把第8个参数设成 1。")
    end
    if search_mode == "layerwise_grid" && n_bo_seeds > 1
        println("  提示: layerwise_grid 的网格是确定性的, 多个种子会重复同样的计算,")
        println("        建议把第8个参数设成 1。(layerwise 用的是随机BO, 多种子仍有意义)")
    end
    println("并行任务数 = 实例数 × BO种子数" * (length(RFS) > 1 ? " × 精修器数" : "") *
            " = $(n_instances*n_bo_seeds*length(RFS)), 可用 worker = $(nworkers())")
    budget_scale != 1.0 &&
        println("⚠ budget_scale=$(budget_scale) —— 这是冒烟测试配置, BO 远未收敛, 结果不能当数据用!")
    bo_verbose > 0 &&
        println("BO 内部过程打印已开启(每 $(bo_verbose) 次迭代一行, 有提升的迭代必打)")
    if n_instances*n_bo_seeds*length(RFS) < nworkers()
        println("提示: 任务数少于 worker 数, 有 worker 会闲置。加大实例数或BO种子数能吃满。")
    end
    println("="^70)

    # ── 先验 ────────────────────────────────────────────────────────────
    prior_tag  = "gnn_prior_target3d_n$(n_at)_full_clean$(target_clean_per_size)_$(optimizer)" *
                 (LGNN_SEPARATE_LINE_WEIGHTS ? "_archfix" : "") * ".jls"
    prior_file = joinpath(directoryBasis, prior_tag)
    if !isfile(prior_file)
        println("没有找到先验, 现在构建训练池并预训练...")
        pool = build_graph_pool_weighted_target3d(n_at, target_clean_per_size; wmin=wmin_fixed, wmax=wmax_fixed)
        println("训练池共 $(length(pool)) 张图")
        params, best_avg = if optimizer == "zygote"
            train_lgnn_pretrain_weighted_zygote(pool; n_epochs=n_epochs, min_epochs=min(2000,n_epochs), batch_size=24)
        else
            train_lgnn_pretrain_weighted(pool; n_epochs=n_epochs, batch_size=24)
        end
        mkpath(directoryBasis); serialize(prior_file, params)
        @printf("✓ 预训练完成, 训练池平均真实离散cut=%.4f\n", best_avg)
    else
        println("复用已缓存的先验 → $(prior_tag)")
    end
    gnn_params = deserialize(prior_file)

    # ── 测试实例: 读实例集文件(QAOA_INSTANCES) 或 照常生成并自动冻结存盘 ──
    # 生成逻辑已抽到 build_test_instances(逐行不变), 见 obtain_test_instances 的说明。
    GRAPHS, INST_FPS, INST_SETFP, INST_FROM_FILE =
        obtain_test_instances(n_at, n_instances, gnn_params; wmin=wmin_fixed, wmax=wmax_fixed,
                              verbose=true, save=true, prior_tag=prior_tag)
    # 读入别处的实例集时, 输出文件名加上实例集指纹, 不覆盖本机自己那批图的结果
    INST_SFX = INST_FROM_FILE ? "I$(INST_SETFP)" : ""
    jtag(parts...) = join(filter(!isempty, collect(parts)), "_")

    # ── 摊平成一个大 pmap: (实例 × BO种子) ──────────────────────────────
    # ── 断点续跑 ────────────────────────────────────────────────────────
    # 每个任务算完立刻单独存盘。作业被 walltime 杀掉 / Jupyter 内核断开 /
    # 手动中断之后, 重跑**同一条命令**会自动跳过已完成的任务, 不用从头再来。
    # 关掉: 设 JULIA_NO_CKPT=1。想强制全部重算: 删掉 ckpt_ 那个目录。
    # 目录名带上 n/p_max/BO种子数/优化器, 不同配置的 checkpoint 不会互相串。
    # [FIX] 目录名里加上 BO 配置指纹。原来只编码 n/p_max/种子数/优化器 ——
    # 结果是把 BO 难度阈值从 dim>=8 改成 dim>=BO_DIM_THRESH 之后, p=3 的配置已经变了,
    # 旧 checkpoint 却照样被静默加载, 拿到的是旧配置算出来的数字, 跟当前代码
    # 不一致而且**没有任何提示**。现在把影响数值结果的几项拼进目录名:
    # 难度阈值(BO_DIM_THRESH)、budget 公式系数、budget_scale。任何一项变了,
    # 目录名就变, 旧结果自动失效, 不会串。
    # [FIX] 逐层类策略原来不把 budget_scale 拼进目录名 —— 冒烟测试(0.05)的结果会
    # 存进跟正式运行同名的目录, 之后正式运行会把这些没收敛的数字当成已完成任务
    # 直接加载。现在 budget_scale≠1 时加后缀; =1 时目录名不变, 旧 checkpoint 照常复用。
    bo_cfg = (startswith(search_mode, "layerwise") || search_mode in ("beam","sweep","hybrid")) ?
             ("$(search_mode)$(layer_param)" * (budget_scale == 1.0 ? "" : "_bs$(budget_scale)") *
              # 束宽也影响结果, 同样拼进去; 为兼容已有的 hybrid 基准目录(束宽5), 束宽=5 时不加
              ((search_mode in ("beam","hybrid") && beam_width_param != 5) ? "_w$(beam_width_param)" : "")) :
             ("d$(BO_DIM_THRESH)b$(BO_BUDGET_A)_$(BO_BUDGET_B)" *
              (budget_scale == 1.0 ? "" : "_bs$(budget_scale)"))
    QAOA_BOX_SCALE != 1.0 && (bo_cfg *= "_box$(QAOA_BOX_SCALE)")   # 参数盒不同 → 结果不同 → 目录隔离
    QAOA_OMEGA_T != 2.0 && (bo_cfg *= "_om$(QAOA_OMEGA_T)")         # 混合层 Ωt 不同 → 曲面不同 → 目录隔离
    WS_MIXSIGN_G < 0 && (bo_cfg *= "_wsE")                           # Egger 精确版 mixer → gnn 曲面不同 → 目录隔离
    WS_EPS_G > 0 && (bo_cfg *= "_ws$(WS_EPS_G)")                     # ε 正则化 → gnn 曲面不同 → 目录隔离
    GP_VERSION >= 2 && (bo_cfg *= "_" * GP_CFG_TAG)                 # GP v2 改变 BO 轨迹 → 目录隔离
    ckpt_dir  = joinpath(directoryBasis,
                         "ckpt_n$(n_at)_p$(p_max)_bs$(n_bo_seeds)_$(optimizer)_$(bo_cfg)")
    # 输出文件名标签: 策略+预算+p_max(+盒)(+外部实例集指纹)。原来论文 CSV 只按
    # n 和实例数命名, 不同配置的运行会互相覆盖(实测: p=1 网格审计把 hybrid 基准的
    # paper_results_n12_inst20.csv 覆盖了)。
    RUN_TAG = jtag("$(bo_cfg)_p$(p_max)", INST_SFX)
    # [QNG] NM 用原目录(名字不变 → 旧 checkpoint 可复用); QNG 单独目录, 目录名带
    # 全部 QNG 超参数, 任何一项改了都自动隔离, 两种精修器的结果绝不会串。
    ckpt_dir_qng = ckpt_dir * "_" * QNG_CFG_TAG
    ckpt_dir_of(rf) = rf == "qng" ? ckpt_dir_qng : ckpt_dir
    use_ckpt  = get(ENV, "JULIA_NO_CKPT", "") != "1"
    if use_ckpt
        for rf in RFS
            mkpath(ckpt_dir_of(rf))
            # 每个 checkpoint 目录记录每个实例的指纹; 换了图(比如读了别的机器的
            # 实例集)还想续用旧目录, 会在这里被拦下, 而不是静默混用两批图的结果
            check_ckpt_fingerprints(ckpt_dir_of(rf), GRAPHS, INST_FPS; from_file=INST_FROM_FILE)
        end
    end
    ckpt_path(inst, s, rf) = joinpath(ckpt_dir_of(rf), "task_i$(inst)_s$(s).jls")

    # 任务序号在构造时就固定 —— worker 打印时才能显示"第几个/共几个"。
    # 没有这个的话日志里只有零散的 inst/boseed, 完全看不出 50 个任务的队列
    # 消化到哪了(实测被误解成"只跑了2个种子", 其实是另外3个还在排队)。
    n_all = length(GRAPHS)*n_bo_seeds*length(RFS)
    # 精修器放在最内层: "both" 时同一 (实例,种子) 的 NM/QNG 两个任务相邻派发,
    # 中途被杀时已完成的部分也是尽量成对的, 对比表能用上更多数据。
    ALL_TASKS = [(g, s, k, n_all, rf)
                 for (k,(g,s,rf)) in enumerate([(gg,ss,rr) for gg in GRAPHS for ss in 1:n_bo_seeds for rr in RFS])]
    DONE  = NamedTuple[]
    TASKS = Tuple[]
    if use_ckpt
        for tk in ALL_TASKS
            g, s, rf = tk[1], tk[2], tk[5]
            f = ckpt_path(g.inst, s, rf)
            if isfile(f)
                try
                    # 旧版 checkpoint 没有 refiner 字段; 这里按它所在的目录补上,
                    # 后面按精修器拆分结果时不会认错
                    push!(DONE, merge(deserialize(f), (refiner=rf,)))
                catch
                    push!(TASKS, tk)   # 文件损坏(多半是写到一半被杀), 重算
                end
            else
                push!(TASKS, tk)
            end
        end
        if !isempty(DONE)
            @printf("断点续跑: 已完成 %d/%d 个任务, 本次只跑剩下的 %d 个\n",
                    length(DONE), n_all, length(TASKS))
            println("  checkpoint 目录: " * join([basename(ckpt_dir_of(rf)) for rf in RFS], " + "))
            isempty(TASKS) && println("  ⚠ 全部任务都来自缓存, 本次没有做任何新计算 —— " *
                                      "结果即刻产出属正常。想强制重算: 删掉上面这个目录, 或设 JULIA_NO_CKPT=1")
        end
    else
        TASKS = ALL_TASKS
    end

    steps_dir = joinpath(directoryBasis, "ProgressSteps")   # 纯 String, 可随闭包序列化到 worker
    mkpath(steps_dir)
    println("\n开始并行计算: $(length(TASKS)) 个任务分发到 $(nworkers()) 个 worker...")
    flush(stdout)
    t_all = time()
    RES_NEW = isempty(TASKS) ? NamedTuple[] : pmap(TASKS) do (g, s, k, ntot, rf)
        # 借鉴范例: 每步都带上主机名和内存 —— 多节点作业时能看出是哪台机器慢,
        # 内存数字则能提前发现某个 worker 在涨(OOM 前的征兆)。
        wlog(@sprintf("    [任务 %d/%d 开始] 实例%d(图seed=%d) BO种子%d 精修器=%s  [host %s | worker %d | mem %.2f GB]",
                      k, ntot, g.inst, g.seed, s, rf, gethostname(), myid(),
                      Sys.maxrss()/2^30), :always)
        r = if search_mode == "layerwise"
                run_one_layerwise_chain(g, s, p_max; use_grid=false,
                                        layer_budget=layer_param, bo_verbose=bo_verbose)
            elseif search_mode == "layerwise_grid"
                run_one_layerwise_chain(g, s, p_max; use_grid=true,
                                        n_grid=layer_param, bo_verbose=bo_verbose)
            elseif search_mode == "beam"
                run_one_layer_strategy_chain(g, s, p_max; strategy=:beam,
                                             budget_scale=budget_scale,
                                             beam_width=beam_width_param, layer_budget=layer_param,
                                             bo_verbose=bo_verbose, refiner=Symbol(rf))
            elseif search_mode == "sweep"
                run_one_layer_strategy_chain(g, s, p_max; strategy=:sweep,
                                             budget_scale=budget_scale,
                                             layer_budget=layer_param,
                                             bo_verbose=bo_verbose, refiner=Symbol(rf))
            elseif search_mode == "hybrid"
                run_one_layer_strategy_chain(g, s, p_max; strategy=:hybrid,
                                             budget_scale=budget_scale,
                                             beam_width=beam_width_param, layer_budget=layer_param,
                                             bo_verbose=bo_verbose, refiner=Symbol(rf))
            else
                run_one_bo_chain(g, s, p_max; budget_scale=budget_scale, bo_verbose=bo_verbose)
            end
        wlog(@sprintf("    [任务 %d/%d 完成] 实例%d BO种子%d 精修器=%s  [耗时 %.1f 分钟 | 峰值内存 %.2f GB | host %s]",
                      k, ntot, g.inst, s, rf,
                      sum(sum(values(r.outt[cd])) for cd in keys(r.outt))/60,
                      Sys.maxrss()/2^30, gethostname()), :always)
        # ── 逐步进度落盘 ────────────────────────────────────────────────
        # 借鉴 ProgressSteps 的做法: 每完成一个任务就在磁盘留一个文件, 文件名
        # 自带序号和完成时刻。好处是 `ls ProgressSteps | wc -l` 一眼看出进度,
        # 不用翻日志; 作业被 walltime 杀掉这些文件也还在, 不像最终 CSV 是跑完
        # 才写。这里不写空文件而是把该任务的全部逐层结果一起写进去 —— 反正
        # 都要落盘, 顺带把数据带上, 中途就能拿来分析。
        # [注意] 这段在 pmap 的 do-block 里, **是在 worker 上执行的**, 所以只能
        # 用字符串拼路径 + 普通 write, 不能碰主进程的 IO 句柄(之前 LOG_FILE_IO[]
        # 那次就是踩了这个坑, 直接 ReadOnlyMemoryError)。
        try
            fn = @sprintf("Step %03d of %03d — inst%d seed%d %s — %s.csv",
                          k, ntot, g.inst, s, rf, Dates.format(now(), "yyyy-mm-dd HH-MM-SS"))
            open(joinpath(steps_dir, fn), "w") do io
                println(io, "p,condition,ar_cmax,ar_sigmaw,elapsed_min")
                for p in 1:p_max, cd in ("cold_start","tqa_init","gnn_warmstart")
                    @printf(io, "%d,%s,%.6f,%.6f,%.2f\n",
                            p, cd, r.out[cd][p], r.outw[cd][p], r.outt[cd][p]/60)
                end
            end
        catch e
            @warn "逐步进度文件写入失败(不影响计算)" exception=e
        end
        # [FIX] 这里**不能**写日志文件。pmap 的 do-block 会被序列化到 worker 上
        # 执行, LOG_FILE_IO[] 是主进程打开的 IOStream —— worker 拿到的是无效副本,
        # 往里 println 直接 ReadOnlyMemoryError。worker 的日志随返回值(logbuf)
        # 带回来, 由主进程在 pmap 结束后统一落盘, 见下面的 flush_worker_logs。
        # (这跟之前 ckpt_path 那次是同一类问题: 要在主进程跑的代码进了闭包。)
        # 算完立刻存盘 —— 之后整个作业就算被杀, 这个任务也不用重算。
        # [FIX] 这里**不能**调主进程定义的 ckpt_path 闭包: pmap 会把 do-block
        # 序列化到 worker 上执行, 而那个函数只在主进程有定义, worker 上会报
        # UndefVarError: #ckpt_path not defined。改成直接用字符串拼路径 ——
        # ckpt_dir 是 String, 能正常随闭包序列化过去。
        if use_ckpt
            try
                # ckpt_dir / ckpt_dir_qng 都是 String, 可随闭包序列化(不能调 ckpt_dir_of 闭包)
                serialize(joinpath(rf == "qng" ? ckpt_dir_qng : ckpt_dir, "task_i$(g.inst)_s$(s).jls"), r)
            catch e
                @warn "checkpoint 写入失败(不影响本次计算)" exception=e
            end
        end
        r
    end
    RES = vcat(DONE, RES_NEW)
    # worker 侧的完整日志(含只进文件的 BO 每步细节)在这里由主进程统一写盘。
    # 中途被杀也不会丢: logbuf 一起存进了 checkpoint, 下次续跑照样能写出来。
    if LOG_FILE_IO[] !== nothing
        # [FIX] 原来在 for 循环里累加 nlines, 而这里是顶层脚本作用域 —— Julia
        # 的软作用域规则会警告"赋值有歧义"。改成先写盘、再用 sum 一次性算出
        # 行数(纯表达式, 不在循环里赋值), 从根上避开这个规则。
        for r in RES
            hasproperty(r, :logbuf) || continue
            for ln in r.logbuf; println(LOG_FILE_IO[], ln); end
        end
        flush(LOG_FILE_IO[])
        nlines = sum(hasproperty(r, :logbuf) ? length(r.logbuf) : 0 for r in RES; init=0)
        @printf("worker 日志已写入文件: %d 行\n", nlines)
    end

    # ── 结构化追踪 CSV ──────────────────────────────────────────────────
    # 每个 (实例, BO种子, p, condition, 阶段) 一条: 时间戳、worker号、耗时、
    # 峰值内存、堆占用、该阶段新增的量子演化次数、当前最优、有没有产生改进。
    # **无改进的阶段也照记** —— 它同样消耗了时间, 是判断"某一步值不值"的关键
    # 依据(比如轮扫到底有没有用, 精修的哪几档扰动是白花的)。
    let traces = [t for r in RES if hasproperty(r, :trace) for t in r.trace]
        if !isempty(traces)
            # nm 时文件名跟旧版一致; qng/both 加后缀, 不覆盖 NM 的 trace
            tf = joinpath(directoryBasis,
                          "trace_n$(n_at)_$(RUN_TAG)" *
                          (refiner_param == "nm" ? "" : "_$(refiner_param)") * ".csv")
            open(tf, "w") do io
                println(io, "timestamp,worker,instance,graph_seed,bo_seed,p,condition," *
                            "stage,ar_cmax,elapsed_s,rss_gb,heap_gb,n_evals,refiner,note")
                for t in traces
                    println(io, join((t.ts, t.worker, t.inst, t.graph_seed, t.bo_seed,
                                      t.p, t.condition, t.stage, round(t.ar_cmax,digits=6),
                                      t.elapsed_s, t.rss_gb, t.heap_gb, t.n_evals,
                                      (hasproperty(t, :refiner) ? t.refiner : "nm"), t.note), ","))
                end
            end
            @printf("✓ 详细追踪 → %s  (%d 条阶段记录)\n", basename(tf), length(traces))
        end
    end
    t_wall = time() - t_all
    @printf("\n并行计算完成, 墙钟 %.1f 分钟\n", t_wall/60)

    # ── 运行时内存峰值实测 ──────────────────────────────────────────────
    # 启动时那次量的是"包加载完"的基线, 不是峰值。Sys.maxrss() 记录的是进程
    # 生命周期内的峰值 RSS, 所以在**计算结束后**再量一次, 拿到的才是标定
    # JULIA_WORKER_GB 该用的数字。
    try
        rss_pk = [remotecall_fetch(() -> Sys.maxrss()/2^30, w) for w in workers()]
        pk_avg = sum(rss_pk)/length(rss_pk); pk_max = maximum(rss_pk)
        @printf("内存峰值实测: 每 worker 平均 %.2f GB, 最大 %.2f GB (主进程 %.2f GB)\n",
                pk_avg, pk_max, Sys.maxrss()/2^30)
        @printf("  → 集群上建议设 JULIA_WORKER_GB=%.1f (按最大值 %.2f 上取整并留 15%% 余量)\n",
                ceil(pk_max*1.15*10)/10, pk_max)
    catch e
        @warn "内存峰值实测失败(不影响结果)" exception=e
    end

    # [QNG] 按精修器拆分出表。nm 单跑时文件名与旧版完全一致(tag 为空)。
    rf_of(r) = hasproperty(r, :refiner) ? r.refiner : "nm"
    if refiner_param == "both"
        RES_NM  = [r for r in RES if rf_of(r) == "nm"]
        RES_QNG = [r for r in RES if rf_of(r) == "qng"]
        mprintln("\n" * "#"^78)
        mprintln("#  论文报表 (A): BO + NelderMead 精修")
        mprintln("#"^78)
        emit_paper_report(GRAPHS, RES_NM, p_max, n_at, n_bo_seeds, optimizer, jtag(RUN_TAG, "nm"))
        mprintln("\n" * "#"^78)
        mprintln("#  论文报表 (B): BO + QNG 精修")
        mprintln("#"^78)
        emit_paper_report(GRAPHS, RES_QNG, p_max, n_at, n_bo_seeds, optimizer, jtag(RUN_TAG, "qng"))
        emit_refiner_comparison(GRAPHS, RES_NM, RES_QNG, p_max, n_at, n_bo_seeds,
                                jtag(RUN_TAG, "bs$(n_bo_seeds)"))
    else
        emit_paper_report(GRAPHS, RES, p_max, n_at, n_bo_seeds, optimizer,
                          jtag(RUN_TAG, refiner_param == "qng" ? "qng" : ""))
    end
    if LOG_FILE_IO[] !== nothing
        flush(LOG_FILE_IO[])
        @printf("日志: 共 %d 行写入文件, 其中 %d 行同时显示在控制台\n",
                LOG_LINES[], CONSOLE_LINES[])
        println("完整细节(含BO每步)在: $(get(ENV,"JULIA_LOG_FILE","(未设置)"))")
    end
    # [FIX] 原来这里引用 CONDS, 但 CONDS 在抽出 emit_paper_report 时被移进了
    # 那个函数, 外层作用域已经没有它了 → UndefVarError。改成直接遍历 outt 的
    # 值, 不依赖外部的 condition 列表。
    let cpu_tot = sum(sum(values(d)) for r in RES for d in values(r.outt); init=0.0)
        @printf("墙钟 %.1f 分钟, CPU 合计 %.1f 分钟 (并行加速 %.1fx, worker=%d)\n",
                t_wall/60, cpu_tot/60, cpu_tot/t_wall, nworkers())
    end
    exit(0)
end

# 单独看每种方案构造出来的图本身健不健康: 嵌入残差、权重范围/比值、
# unwanted_ratio。用来快速判断"这个方案适不适合加权随机3-regular图",不用
# 每次都跑一遍完整的 gnn_qaoa_weighted 才能看结果。
#
# 运行方式:
#   julia r3g_sge.jl weighted_geom_compare <n_atoms> [seed] [d_fixed]
# 例如:
#   julia r3g_sge.jl weighted_geom_compare 10 42 14.0
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "weighted_geom_compare"
    n_at    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    seed    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 42
    d_fixed = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 14.0
    using Statistics, Printf

    println("\n" * "="^70)
    println("加权随机3-regular图: 构造方式对比 (n=$(n_at), seed=$(seed), d=$(d_fixed))")
    println("="^70)

    results = NamedTuple[]

    for km in ["global", "legacy"]
        println("\n── force ($(km)) ──")
        atoms_f, edges_f, weights_f, stress_f, unwanted_f =
            make_reg3_weighted_graph_force(n_at, seed; k_mode=km, d_target=d_fixed)
        push!(results, (name="force-$(km)", stress=stress_f,
                         wmin=minimum(weights_f), wmax=maximum(weights_f),
                         wratio=maximum(weights_f)/minimum(weights_f), unwanted=unwanted_f, fidelity=NaN))
    end

    println("\n── jitter (拓扑=蜂窝格, 抖动=15%) ──")
    atoms_j, edges_j, weights_j, unwanted_j =
        make_honeycomb_graph_jittered(n_at, d_fixed; jitter_frac=0.15, seed=seed)
    push!(results, (name="jitter-0.15", stress=0.0,
                     wmin=minimum(weights_j), wmax=maximum(weights_j),
                     wratio=maximum(weights_j)/minimum(weights_j), unwanted=unwanted_j, fidelity=NaN))

    println("\n── greedy (贪心最近邻, 拓扑真随机) ──")
    atoms_g, edges_g, weights_g, unwanted_g =
        make_reg3_graph_geometric_greedy(n_at, d_fixed, seed)
    push!(results, (name="greedy", stress=0.0,
                     wmin=minimum(weights_g), wmax=maximum(weights_g),
                     wratio=maximum(weights_g)/minimum(weights_g), unwanted=unwanted_g, fidelity=NaN))

    println("\n── force3d (legacy, 三维) ──")
    atoms_f3, edges_f3, weights_f3, stress_f3, unwanted_f3 =
        make_reg3_weighted_graph_force_3d(n_at, seed; d_target=d_fixed)
    push!(results, (name="force3d-legacy", stress=stress_f3,
                     wmin=minimum(weights_f3), wmax=maximum(weights_f3),
                     wratio=maximum(weights_f3)/minimum(weights_f3), unwanted=unwanted_f3, fidelity=NaN))

    println("\n── greedy3d (贪心最近邻, 三维) ──")
    atoms_g3, edges_g3, weights_g3, unwanted_g3 =
        make_reg3_graph_geometric_greedy_3d(n_at, d_fixed, seed)
    push!(results, (name="greedy3d", stress=0.0,
                     wmin=minimum(weights_g3), wmax=maximum(weights_g3),
                     wratio=maximum(weights_g3)/minimum(weights_g3), unwanted=unwanted_g3, fidelity=NaN))

    println("\n── target3d (逐边目标权重, 三维力场) ──")
    atoms_t3, edges_t3, weights_t3, stress_t3, unwanted_t3, fidelity_t3 =
        make_reg3_weighted_graph_target3d(n_at, seed)
    push!(results, (name="target3d", stress=stress_t3,
                     wmin=minimum(weights_t3), wmax=maximum(weights_t3),
                     wratio=maximum(weights_t3)/minimum(weights_t3), unwanted=unwanted_t3, fidelity=fidelity_t3))

    println("\n" * "="^70)
    println("汇总对比")
    println("="^70)
    println("  方案            嵌入残差        权重范围                权重比值    unwanted_ratio   目标权重相关系数")
    println("  " * "─"^96)
    for r in results
        unwanted_str = isnan(r.unwanted) ? "N/A" : string(round(r.unwanted, digits=6))
        fidelity_str = isnan(r.fidelity) ? "N/A" : string(round(r.fidelity, digits=4))
        @printf("  %-14s  %10.3f    [%.5f, %.5f]   %10.2f    %-14s  %s\n",
                r.name, r.stress, r.wmin, r.wmax, r.wratio, unwanted_str, fidelity_str)
    end
    println("  " * "─"^96)
    println("  jitter/greedy(2d/3d)的\"嵌入残差\"恒为0(没有优化步骤, 这个概念本来就")
    println("  不适用, 打0只是为了跟其它模式统一格式); unwanted_ratio 应远小于1;")
    println("  目标权重相关系数只有target3d才有意义, 越接近1说明实际权重跟最初")
    println("  设定的目标权重越吻合; 权重比值越接近个位数越健康。")
    println("="^70)
    exit(0)
end


# 跟 gdqc_validate 结构一致, 三处替换:
#   1. 图集合换成多张不同大小/种子的带权 3-regular 图 (make_reg3_weighted_graph
#      + embed_weighted_graph_2d)
#   2. GNN 训练用 train_lgnn_quick_weighted
#   3. "自身最优角度"/"零样本迁移"评估全部用带权版本 (quick_best_angles_weighted,
#      random_angle_baseline_weighted, loss_fn_weighted)
# χ(G) 的距离轴保持拓扑距离不变 (不受权重影响) —— 这是刻意的, 跟聊天记录里
# 讨论过的一样: GDQC 的距离轴设计上编码节点属性/拓扑关系, 不是边权重。
#
# 运行方式:
#   julia r3g_sge.jl gdqc_validate_weighted <p_val> [n_c_bins]
# 例如:
#   julia r3g_sge.jl gdqc_validate_weighted 3 10
# ══════════════════════════════════════════════════════════════════════════

if length(ARGS) >= 1 && ARGS[1] == "gdqc_validate_weighted"
    using Statistics, LinearAlgebra, Printf

    p_val    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    n_c_bins = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10

    println("\n" * "="^70)
    println("GDQC 验证实验 (加权图): 核相似度是否预测 QAOA 角度可迁移性")
    println("p_val = $(p_val)  (dim = $(2*p_val)D)   n_c_bins = $(n_c_bins)")
    println("="^70)

    # 待比较的带权 3-regular 图集合: 不同大小 + 不同种子。
    WEIGHTED_GRAPH_SPECS = [
        (8,  1),
        (8,  2),
        (10, 1),
        (10, 2),
        (12, 1),
        (12, 2),
    ]

    println("\n[Step ①+②] 为每张带权图: 训练 LGNN → 提取 GDQC 特征 χ(G) → 求自身最优角度\n")

    entries = GDQCLibEntryWeighted[]
    for (n_at, sd) in WEIGHTED_GRAPH_SPECS
        # [FIX] 原来用 make_reg3_weighted_graph(target方式: 先定权重再嵌入)
        # —— 已确认对随机3-regular拓扑残差大到能让权重方向反转, 不再使用。
        # 换成 make_reg3_weighted_graph_force(legacy k_mode, 三轮实测里最
        # 稳的方案)。
        atoms, edges, weights, stress_val, _ =
            make_reg3_weighted_graph_force(n_at, sd; k_mode="legacy", d_target=14.0)
        n_atoms = n_at; n_edges = length(edges)
        label = "reg3w(n=$(n_atoms),seed=$(sd))"
        print("  $(label)  |V|=$(n_atoms) |E|=$(n_edges) stress=$(round(stress_val,digits=3))  ... ")
        flush(stdout)

        gnn_file = joinpath(directoryBasis, "gnn_reg3w_n$(n_atoms)_seed$(sd).jls")
        local p_gnn
        if isfile(gnn_file)
            _, p_gnn, _, _, _, _ = deserialize(gnn_file)
        else
            gnn_params_, p_gnn, gnn_ar = train_lgnn_quick_weighted(edges, weights, n_atoms; n_epochs=300)
            mkpath(directoryBasis)
            serialize(gnn_file, (gnn_params_, p_gnn, gnn_ar, n_atoms, n_edges, edges))
        end

        chi = gdqc_signature(p_gnn, atoms, edges, n_atoms; n_c_bins=n_c_bins)
        best_bx, best_ar = quick_best_angles_bo_weighted(atoms, edges, weights, n_atoms, p_val)
        cold_ar = random_angle_baseline_weighted(atoms, edges, weights, n_atoms, p_val; n_samples=20)
        _, gnn_solve_ar = quick_best_angles_bo_gnn_weighted(atoms, edges, weights, n_atoms, p_gnn, p_val)

        @printf("own_ar=%.4f  cold_ar=%.4f  gnn_ar=%.4f  |χ|=%d\n",
                best_ar, cold_ar, gnn_solve_ar, length(chi))
        push!(entries, GDQCLibEntryWeighted(label, n_atoms, label, n_edges, chi,
                                             best_bx, best_ar, cold_ar, gnn_solve_ar,
                                             atoms, edges, weights))
    end

    mkpath(directoryBasis)
    lib_path = joinpath(directoryBasis, "gdqc_library_weighted_p$(p_val).jls")
    serialize(lib_path, entries)
    println("\n✓ 核库已保存 → $(lib_path)")

    println("\n[Step ②] 零样本角度迁移实验 (加权): 遍历所有有序对 (A→B)\n")

    rows = NamedTuple[]
    for A in entries, B in entries
        A === B && continue
        kappa = kappa_gdqc(A.chi, B.chi)
        lv = try
            loss_fn_weighted(A.best_bx, B.atoms, B.edges, B.weights, B.n_atoms)
        catch
            1000.0
        end
        total_w_B = sum(B.weights)
        ar_transfer = isfinite(lv) ? -clamp(lv, -(total_w_B+1.0), 1000.0)/total_w_B : 0.0
        gain = ar_transfer - B.cold_ar
        push!(rows, (A=A.label, B=B.label, kappa=kappa, ar_transfer=ar_transfer,
                      cold_ar_B=B.cold_ar, gain=gain, own_ar_A=A.best_ar, own_ar_B=B.best_ar,
                      gnn_ar_A=A.gnn_ar, gnn_ar_B=B.gnn_ar))
    end

    csv_path = joinpath(directoryBasis, "gdqc_validation_pairs_weighted_p$(p_val).csv")
    open(csv_path, "w") do io
        println(io, "graph_A,graph_B,kappa_gdqc,ar_transfer,cold_ar_B,transfer_gain,own_ar_A,own_ar_B,gnn_ar_A,gnn_ar_B")
        for r in rows
            @printf(io, "%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.A, r.B, r.kappa, r.ar_transfer, r.cold_ar_B, r.gain, r.own_ar_A, r.own_ar_B,
                    r.gnn_ar_A, r.gnn_ar_B)
        end
    end
    println("✓ 明细表已保存 → $(csv_path)")

    kappas = [r.kappa for r in rows]
    gains  = [r.gain  for r in rows]
    pearson_r = length(rows) >= 3 ? cor(kappas, gains) : NaN

    println("\n" * "="^70)
    println("验证结果汇总 (加权图, 共 $(length(rows)) 个有序图对)")
    println("="^70)
    @printf("  κ_GDQC 范围     : [%.4f, %.4f]\n", minimum(kappas), maximum(kappas))
    @printf("  transfer_gain   : mean=%.4f  std=%.4f  (>0 表示比随机基线好)\n",
            mean(gains), std(gains))
    @printf("  Pearson r(κ, gain) = %.4f\n", pearson_r)
    println("─"^70)

    if pearson_r > 0.4 && mean(gains) > 0
        println("  判定: 验证通过 ✓ —— 加权图上 κ_GDQC 与迁移增益仍然正相关。\n")
    elseif pearson_r > 0.15
        println("  判定: 弱相关 —— 建议扩大 WEIGHTED_GRAPH_SPECS 的图数量再验证。\n")
    else
        println("  判定: 未通过验证 —— 加权图上 κ_GDQC 不能可靠预测角度可迁移性,")
        println("        跟无权图的验证结果可能不同, 请对比 gdqc_validate 的结果。\n")
    end
    println("="^70)
    exit(0)
end
