# 1. Installing Docker (Windows, macOS & Linux)

You only do this **once** per computer. Take your time — read every step.

> **What is Docker?** Think of it as a "computer inside your computer". We ship
> one box (an *image*) that already has every tool, library and the exact
> versions our code needs. You run that box (a *container*) and everything just
> works — no "it works on my machine" problems. You keep editing code in your
> normal editor on your real computer; the container only *runs* it.

---

## A. Windows (with an NVIDIA GPU) — full GPU support 🚀

Your GPU works inside Docker on Windows through **WSL2** (a small Linux that
Windows runs for you). Docker Desktop sets most of this up automatically.

### Step 1 — Update your NVIDIA driver
1. Open **GeForce Experience** (or go to <https://www.nvidia.com/download/index.aspx>).
2. Install the latest **Game Ready** or **Studio** driver. Reboot.
   - You do **not** install CUDA on Windows — the driver is enough. CUDA lives
     inside the container.

### Step 2 — Turn on WSL2
1. Open **PowerShell as Administrator** (right-click → *Run as administrator*).
2. Run:
   ```powershell
   wsl --install
   ```
3. Reboot when it asks. After reboot it may ask you to create a Linux username
   and password — pick anything you'll remember.
4. Verify:
   ```powershell
   wsl --status
   ```
   It should say *Default Version: 2*.

### Step 3 — Install Docker Desktop
1. Download from <https://www.docker.com/products/docker-desktop/> (Windows).
2. Run the installer; **keep "Use WSL 2 instead of Hyper-V" checked**.
3. Start **Docker Desktop**. Wait until the whale icon in the taskbar is steady
   (not animating).
4. In Docker Desktop → **Settings → Resources → WSL Integration**, make sure
   integration with your Linux distro is **ON**.

### Step 4 — Check the GPU is visible to Docker
Open **PowerShell** (normal, not admin) and run:
```powershell
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```
✅ If you see a table listing your GPU — you're done, GPU works.
❌ If you see an error about "could not select device driver", re-check Step 1
   and that Docker Desktop is running, then reboot once more.

> **Where should the project live?** Put this project **inside WSL**, not on the
> Windows `C:` drive, or file access is very slow. Open the **Ubuntu** app from
> the Start menu and work there (your files are at `\\wsl$\Ubuntu\home\...` if
> you want to see them in Windows Explorer). Use VS Code with the **WSL**
> extension to edit them comfortably.

---

## B. macOS — CPU only (read this carefully) 🐢

**Important truth:** Docker on a Mac **cannot use an NVIDIA GPU**. Macs don't
have NVIDIA GPUs and macOS has no CUDA. So on your Mac everything runs on the
**CPU**. That is perfectly fine for:
- writing and debugging code,
- running the MuJoCo / PyBullet simulators,
- small/quick training runs.

It is **too slow for big training runs**. For those, SSH into a machine that has
an NVIDIA GPU (a Windows/Linux workstation or a remote GPU server) and run the
exact same commands there — the Docker setup is identical.

### Step 1 — Install Docker Desktop
1. Download from <https://www.docker.com/products/docker-desktop/>.
   - **Apple Silicon (M1/M2/M3/M4)** → choose the *Apple chip* build.
   - **Intel Mac** → choose the *Intel chip* build.
2. Open the `.dmg`, drag Docker to *Applications*, launch it, finish the setup.
3. Wait until the whale icon in the menu bar is steady.

### Step 2 — (Apple Silicon only) allow x86 images
Our image is built for Intel/AMD (`linux/amd64`) because CUDA only exists for
that architecture. Apple Silicon can still run it through emulation (slower).
Turn emulation on:
- Docker Desktop → **Settings → General** → enable
  **"Use Rosetta for x86/amd64 emulation"** (and *Apply & Restart*).

You'll also tell Docker to use that platform when building — see
[2-build-and-run.md](2-build-and-run.md), the macOS note.

### Step 3 — Check Docker works
```bash
docker run --rm hello-world
```
✅ If you see "Hello from Docker!" you're ready. (Do **not** add `--gpus all` on
a Mac — it will error; there's no GPU to give.)

---

## C. Linux (with an NVIDIA GPU) — full GPU support 🚀

On Linux the GPU works in Docker natively — no WSL needed.

### Step 1 — Install the NVIDIA driver
Use your distro's driver (e.g. on Ubuntu: *Software & Updates → Additional
Drivers*, or `sudo ubuntu-drivers autoinstall`), then reboot. Check with
`nvidia-smi`. You do **not** need to install CUDA on the host — it's in the image.

### Step 2 — Install Docker Engine
Follow <https://docs.docker.com/engine/install/> for your distro. Then add
yourself to the `docker` group so you don't need `sudo`:
```bash
sudo usermod -aG docker $USER   # then log out and back in
```

### Step 3 — Install the NVIDIA Container Toolkit
This is what lets Docker hand the GPU to containers. Follow
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>,
then:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 4 — Check the GPU is visible to Docker
```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```
✅ A table listing your GPU means you're done.

---

## What you need on ALL systems before continuing
- **Git** (to download the code repos):
  - Windows (inside WSL/Ubuntu): `sudo apt install git`
  - macOS: it comes with the *Xcode Command Line Tools*; if missing, run
    `xcode-select --install`.
  - Linux: `sudo apt install git` (or your distro's package manager).

➡️ Next: [2-build-and-run.md](2-build-and-run.md)
