import React, { useState } from "react";

const styles = {
  body: {
    fontFamily: "'Segoe UI', sans-serif",
    backgroundColor: "#f9fdfb",
    color: "#333",
    margin: 0,
    padding: 0,
  },
  header: {
    background: "linear-gradient(to right, #d4f1d4, #f0fdf0)",
    padding: "2rem",
    textAlign: "center",
  },
  headerH1: {
    fontSize: "2.5rem",
    color: "#2c6e49",
    marginBottom: "0.5rem",
  },
  headerP: {
    fontSize: "1.2rem",
    color: "#4a7c59",
  },
  section: {
    padding: "2rem",
    maxWidth: "800px",
    margin: "auto",
  },
  sectionH2: {
    color: "#2c6e49",
    marginBottom: "1rem",
  },
  questionnaire: {
    backgroundColor: "#ffffff",
    borderRadius: "8px",
    padding: "1.5rem",
    boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
  },
  questionGroup: {
    marginBottom: "2rem",
  },
  questionGroupH3: {
    color: "#3b7a57",
    marginBottom: "1rem",
  },
  question: {
    marginBottom: "1rem",
  },
  questionLabel: {
    display: "block",
    marginBottom: "0.5rem",
    fontWeight: 500,
  },
  options: {
    display: "flex",
    gap: "1rem",
    flexWrap: "wrap",
  },
  optionLabel: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    fontWeight: "normal",
  },
  submitButton: {
    backgroundColor: "#2c6e49",
    color: "#fff",
    border: "none",
    padding: "0.75rem 1.5rem",
    fontSize: "1rem",
    borderRadius: "4px",
    cursor: "pointer",
  },
  resultBox: {
    marginTop: "1rem",
    padding: "1rem",
    borderRadius: "6px",
  },
  success: {
    backgroundColor: "#d4edda",
    color: "#155724",
  },
  warning: {
    backgroundColor: "#fff3cd",
    color: "#856404",
  },
  danger: {
    backgroundColor: "#f8d7da",
    color: "#721c24",
  },
  secondary: {
    backgroundColor: "#e2e3e5",
    color: "#6c757d",
  },
};

const questionGroups = [
  {
    title: "Part 1: Social Interaction & Communication",
    questions: [
      "Does your child respond to their name when called?",
      "Does your child make eye contact when talking or playing with others?",
      "Does your child smile back when you smile at them?",
      "Does your child look at objects when you point to them? (Joint attention)",
      "Does your child use gestures (e.g., waving, pointing, reaching) to communicate?",
      "Does your child show interest in playing with other children?",
      "Does your child bring objects to show you just for sharing (not for help)?",
      "Does your child imitate your actions (e.g., clapping, waving, making faces)?",
      "If you point at something across the room, does your child look at it?",
      "Does your child try to get your attention by making sounds or gestures?",
    ],
  },
  {
    title: "Part 2: Speech, Language & Cognitive Skills",
    questions: [
      "Does your child use single words (e.g., \"mama,\" \"ball\") by 12–16 months?",
      "By 24 months, does your child combine two words (e.g., “want cookie”)?",
      "Does your child babble or make different sounds to express emotions?",
      "Does your child understand simple instructions (e.g., “Give me the toy”)?",
      "Does your child follow eye gaze (e.g., looking where you look)?",
      "Does your child struggle to play pretend games (e.g., feeding a doll)?",
      "Does your child repeat words or phrases (echolalia) instead of using them meaningfully?",
      "Has your child ever lost language or social skills they previously had?",
      "Does your child seem uninterested in imitating speech or sounds?",
      "Does your child use their voice in unusual ways (e.g., monotone, robotic, singsong)?",
    ],
  },
  {
    title: "Part 3: Repetitive Behaviors & Restricted Interests",
    questions: [
      "Does your child flap their hands, rock back and forth, or spin repetitively?",
      "Does your child insist on doing things the same way every time (e.g., same routine, same path)?",
      "Does your child show an intense attachment to specific objects (e.g., a toy, bottle, or part of an object like a wheel)?",
      "Does your child play with toys in an unusual way (e.g., lining up, spinning wheels, fixating on parts)?",
      "Does your child react strongly to small changes in routine or environment?",
      "Does your child repeat the same action over and over without a clear purpose?",
      "Does your child seem overly interested in numbers, letters, or patterns?",
      "Does your child show unusual fascination with certain sounds, movements, or lights?",
      "Does your child stare at moving objects (e.g., fans, spinning wheels) for a long time?",
      "Does your child engage in self-soothing behaviors (e.g., head-banging, biting, or repetitive touching)?",
    ],
  },
  {
    title: "Part 4: Sensory Sensitivities & Motor Skills",
    questions: [
      "Does your child overreact to certain sounds (e.g., vacuum, sirens, loud music)?",
      "Does your child dislike certain textures (e.g., clothes, food, surfaces)?",
      "Does your child show little or no reaction to pain or extreme temperatures?",
      "Does your child seem fascinated with certain sensory experiences (e.g., feeling soft fabrics, rubbing surfaces)?",
      "Does your child show difficulty with fine motor skills (e.g., holding a spoon, grasping small objects)?",
      "Does your child walk on their toes often?",
      "Did your child experience delays in crawling, walking, or sitting up?",
      "Does your child have unusual body posture or clumsiness?",
      "Does your child struggle with hand-eye coordination (e.g., stacking blocks, picking up small objects)?",
      "Does your child seem unaware of personal space, getting too close or avoiding touch?",
    ],
  },
];

const AutismScreeningForm = () => {
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ answers: formData })
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.interpretation);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.body}>
      <header style={styles.header}>
        <h1 style={styles.headerH1}>Autism Spectrum Disorder</h1>
        <p style={styles.headerP}>Early Autism Screening for Toddlers</p>
      </header>

      <section style={styles.section}>
        <h2 style={styles.sectionH2}>Autism Screening Questionnaire</h2>
        <form style={styles.questionnaire} id="autismForm" onSubmit={handleSubmit}>
          {questionGroups.map((group, gIdx) => (
            <div key={gIdx} style={styles.questionGroup}>
              <h3 style={styles.questionGroupH3}>{group.title}</h3>
              {group.questions.map((question, qIdx) => {
                const questionNumber = gIdx * 10 + qIdx + 1;
                const inputName = `q${questionNumber}`;
                return (
                  <div key={inputName} style={styles.question}>
                    <label style={styles.questionLabel} htmlFor={inputName}>
                      {questionNumber}. {question}
                    </label>
                    <div style={styles.options}>
                      <label style={styles.optionLabel}>
                        <input
                          type="radio"
                          name={inputName}
                          value="1"
                          required
                          onChange={handleChange}
                          checked={formData[inputName] === "1"}
                        />
                        Yes
                      </label>
                      <label style={styles.optionLabel}>
                        <input
                          type="radio"
                          name={inputName}
                          value="0"
                          onChange={handleChange}
                          checked={formData[inputName] === "0"}
                        />
                        No
                      </label>
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
          <button type="submit" style={styles.submitButton} disabled={loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>
        {error && <div style={{ ...styles.resultBox, ...styles.secondary }}>{error}</div>}
        {result && (
          <div style={{ ...styles.resultBox, ...styles[result.color] }}>
            <h3>{result.level} Risk</h3>
            <p>{result.message}</p>
          </div>
        )}
      </section>
    </div>
  );
};

export default AutismScreeningForm;
