# Lesson 1 — What is Reinforcement Learning?

⏱️ ~20 minutes of reading and thinking. No coding yet.

---

## 1. The puppy and the treat

Imagine teaching a puppy to sit. You can't explain it with words. Instead:

* The puppy **tries** something (sits, barks, rolls over).
* If it does the right thing, it gets a **treat**.
* Over many tries, it does the treat-earning thing more often.

That's **Reinforcement Learning (RL)**. We replace the puppy with a computer
program called an **agent**, and the treat with a number called a **reward**.

> **RL in one sentence:** an agent learns to act by trying things and keeping
> the actions that earn the most reward.

---

## 2. The four words you must know

```
        ┌──────────────────────────────────────────────┐
        │                                                │
        ▼                                                │
   ┌─────────┐         ACTION                  ┌──────────────┐
   │  AGENT  │ ───────────────────────────────▶│ ENVIRONMENT  │
   │         │                                 │              │
   │         │◀─────────────────────────────── │              │
   └─────────┘   OBSERVATION  +  REWARD        └──────────────┘
```

| Word | Puppy version | Our drone version |
|------|---------------|-------------------|
| **Agent** | the puppy's brain | a small neural network |
| **Environment** | the living room | the drone + the physics simulator |
| **Observation** | what the puppy sees/smells | where the drone is, how fast, how tilted |
| **Action** | sit / bark / roll | how to tilt and how hard to push the motors |
| **Reward** | a treat (or no treat) | a number: high when hovering on target |

One **step** = the agent looks (observation), acts (action), and gets graded
(reward). One **episode** = many steps from start until the drone crashes or
time runs out. Training = **millions** of steps.

---

## 3. Why is this hard? (The credit problem)

If the drone crashes after 3 seconds, *which* of the 150 little decisions it
made caused the crash? Maybe the bad move was 2 seconds ago, not right before
the crash. Figuring out which actions deserve credit (or blame) is the central
puzzle of RL. The algorithm we use (**PPO**) is a clever, stable way to solve
it. You'll meet it in Lesson 3 and look inside it in Lesson 6.

---

## 4. Why hovering?

Hovering looks boring, but it is the "hello world" of drone control:

* It is **simple to describe**: stay at one point, stay still.
* It is **hard to fake**: the drone is always falling, so the agent must
  constantly push back. There is no "do nothing" answer.
* Everything else (following a path, landing, racing) is built on top of it.

The famous [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones)
project starts the same way.

---

## 5. Our drone and our world

* **Drone:** the Crazyflie 2.1 Brushless — a real, palm-sized quadrotor
  (called `cf21B_500` in the code).
* **Simulator:** [Crazyflow](https://github.com/learnsyslab/crazyflow), which
  runs the real physics of the drone on the computer — and can simulate
  *hundreds of drones at once* so training is fast.

We train in simulation because a real drone would crash thousands of times
while learning. In the simulator, crashing is free.

---

## ✅ Check your understanding

1. In our project, what is the *agent*? What is the *environment*?
2. What makes the reward *high*? What makes it *low*?
3. Why do we train in a simulator instead of on a real drone?
4. What is the difference between a *step* and an *episode*?

When you can answer these, go to **Lesson 2** to see how we describe the hover
task to the computer.
